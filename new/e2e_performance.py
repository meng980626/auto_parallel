import pandas as pd
import joblib

CACHE_block = {}
CACHE_other = {}

def mb_time(stage_index, config, device, xgb_model, label_model):
    stage_layer_number = config['num_layers'] // config['PP_size']
    # last stage
    if stage_index == config['PP_size'] - 1 and config['num_layers'] % config['PP_size'] != 0:
        stage_layer_number = config['num_layers'] % config['PP_size']
    cat_cols = ['Rank','device type','recompute granularity']

    single = {
        'TP size': config['TP_size'],
        'Rank': 0,
        'device type': device,
        'max position embedding': config['max_position_embedding'],
        'micro batch size': config['micro_batch_size'],
        'num attention heads': config['num_attention_heads'],
        'num layers': 1,         
        'num query groups': config['num_query_groups'], 
        'recompute granularity': config['recompute_granularity'],
        'sequence length': config['sequence_length'],
    }

    single_key = tuple(sorted(single.items()))
    
    if single_key in CACHE_block:
        return CACHE_block[single_key]*stage_layer_number, CACHE_other[single_key]

    double = single.copy()
    double['num layers'] = 2
    single_df = pd.DataFrame([single])  
    double_df = pd.DataFrame([double])      
    for col in label_model.keys():
        le = label_model[col]
        single_df[col] = le.transform(single_df[col])
        double_df[col] = le.transform(double_df[col])
    pred = xgb_model.predict(single_df)
    pred2 = xgb_model.predict(double_df)
    if pred[0] < 0 or pred2[0] < 0:
        print(config)
    
    CACHE_block[single_key] = (pred2[0] - pred[0]) if (pred2[0] - pred[0]) > 0 else pred[0] * 0.5
    CACHE_other[single_key] = (pred[0] - CACHE_block[single_key]) if (pred[0] - CACHE_block[single_key]) > 0 else pred[0] * 0.5

    # assert CACHE_block[single_key] > 0, f"1-layer time cannot be negative. CACHE_block[single_key]:{CACHE_block[single_key]} pred2[0]:{pred2[0]} pred[0]:{pred[0]} device:{device} config:{config}"
    # assert CACHE_other[single_key] > 0, f"pred2[0]:{pred2[0]}, pred[0]:{pred[0]} other fixed overhead cannot be negative. device:{device} config:{config}"
    
    return CACHE_block[single_key]*stage_layer_number, CACHE_other[single_key]



#device是一个长度等于pp的list，记录各个stage的device信息，bandwidth是一个长度为pp-1的list，记录stage间的通信带宽
def amp_e2e_time(config, device, pp_bandwidth, dp_bandwidth, model, label_model):

    gas = config['global_batch_size'] // config['DP_size'] // config['micro_batch_size'] 

    mb_times = []
    fix_overhead = -1

    for i in range(config['PP_size']):
        mb_time_i, fix_overhead_i = mb_time(i, config, device[i], model, label_model)
        mb_times.append(mb_time_i)
        if fix_overhead_i > fix_overhead:
            fix_overhead = fix_overhead_i

    pp_comm_times = [config['micro_batch_size'] * config['sequence_length'] * config['hidden_state'] * config['dtype_bytes'] / pp_bandwidth[i] for i in range(len(pp_bandwidth))]

    t_pp = (gas - 1) * max(mb_times) + sum(mb_times) + 2 * sum(pp_comm_times) + fix_overhead

    M = 2 * config['vocab_size'] * config['hidden_state'] + config['num_layers'] * (4 * config['hidden_state'] * config['hidden_state'] + 2 * config['hidden_state'] * config['ffn_hidden_state']) * config['dtype_bytes']

    dp_time = 2 * (config['DP_size'] - 1) * M / dp_bandwidth / config['DP_size']

    return t_pp + dp_time


def pipette_e2e_time(config, device, bandwidth):
    
    rf_loaded = joblib.load('xgb_time.pkl')

    mb_times = []

    gas = config.global_batch_size // config.DP_size // config.micro_batch_size

    for i in range(config.PP_size):
        mb_times.append(mb_time(i, config, device[i], rf_loaded))

    pp_comm_times = [config.micro_batch_size * config.sequence_length * config.hidden_state * config.dtype_bytes / bandwidth[i] for i in range(len(bandwidth))]

    t_bubble = sum(mb_times) + 2 * sum(pp_comm_times)

    t_straggler = (config.PP_size - 1) * max(mb_times)

    M = 2 * config.vocab_size * config.hidden_state + config.num_layers * (4 * config.hidden_state * config.hidden_state + 2 * config.hidden_state * config.ffn_hidden_state) * config.dtype_bytes

    dp_time = 2 * (config.DP_size - 1) * M / min(bandwidth) / config.DP_size

    return t_bubble * gas / config.PP_size + t_straggler + dp_time

#相比amp和metis仅仅是考虑了通信延迟
def hexiscale_e2e_time(config, device, bandwidth, latency):
    return amp_e2e_time(config, device, bandwidth) + sum(latency) + max(latency)

#根据vp的数量减少
def interleave_e2e_time(config, device, bandwidth):
    rf_loaded = joblib.load('xgb_time.pkl')

    gas = config.global_batch_size // config.DP_size // config.micro_batch_size 

    mb_times = []

    for i in range(config.PP_size):
        mb_times.append(mb_time(i, config, device[i], rf_loaded))

    pp_comm_times = [config.micro_batch_size * config.sequence_length * config.hidden_state * config.dtype_bytes / bandwidth[i] for i in range(len(bandwidth))]

    t_pp = (gas - 1 + 1/config.vp_size) * max(mb_times) + sum(mb_times)/config.vp_size + 2*sum(pp_comm_times)

    M = 2 * config.vocab_size * config.hidden_state + config.num_layers * (4 * config.hidden_state * config.hidden_state + 2 * config.hidden_state * config.ffn_hidden_state) * config.dtype_bytes

    dp_time = 2 * (config.DP_size - 1) * M / min(bandwidth) / config.DP_size

    return t_pp + dp_time

def chimera_e2e_time(config, device, bandwidth):

    C_f = config.global_batch_size // config.DP_size // config.micro_batch_size 

    assert C_f % config.PP_size == 0

    C_d = C_f / config.PP_size * (2 * config.PP_size - 2)

    pp_comm_times = [config.micro_batch_size * config.sequence_length * config.hidden_state * config.dtype_bytes / bandwidth[i] for i in range(len(bandwidth))]

    mb_times = []

    for i in range(config.PP_size):
        mb_times.append(mb_time(i, config, device[i], rf_loaded))

    mb_fts = [mb_times[i] / device[i].fb_ratio for i in range(len(mb_times))]
    mb_bts = [mb_times[i] * (1 - 1 / device[i].fb_ratio) for i in range(len(mb_times))]

    t_pp = sum(mb_fts) + (C_f - config.PP_size) * max(mb_fts) + sum(mb_bts) + (C_d - config.PP_size) * max(mb_bts) + 2 * sum(pp_comm_times) + sum(pp_comm_times[:(C_d - config.PP_size)])

    M = 2 * config.vocab_size * config.hidden_state + config.num_layers * (4 * config.hidden_state * config.hidden_state + 2 * config.hidden_state * config.ffn_hidden_state) * config.dtype_bytes

    dp_time = 2 * (config.DP_size - 1) * M / min(bandwidth) / config.DP_size

    return t_pp + dp_time