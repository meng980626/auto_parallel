import e2e_performance
import joblib
import itertools
import time

start_time = time.time()
TOTAL_GPU = 4

VALID_PARALLEL = []

for tp in range(1, TOTAL_GPU + 1):
    for pp in range(1, TOTAL_GPU // tp + 1):
        dp = TOTAL_GPU // (tp * pp)
        if tp * pp * dp == TOTAL_GPU:
            VALID_PARALLEL.append((tp, pp, dp))

GLOBAL_BATCH_SIZE = 32
FFN_HIDDEN_SIZE = 11008
HIDDEN_SIZE = 4096
VOCAB_SIZE = 32000
CP_SIZE = 1
DTYPE = ['fp16']
RECOMPUTE_GRANULARITY = ['full','selective','null']
RECOMPUTE_METHOD = ['uniform','block']

param_grid = {
    "NUM_LAYERS": [1,2,4,8,16,32,64],
    "SEQ_LENGTH": [512,1024,2048],
    "MICRO_BATCH_SIZE": [x for x in range(2, GLOBAL_BATCH_SIZE) if (x & (x - 1)) == 0],
    "MAX_POSITION_EMBEDDINGS": [2048,4096,8192],
    "NUM_ATTENTION_HEADS": [1,2,4,8,16,32],
    "NUM_QUERY_GROUPS": [1,2,4,8,16,32],
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
predictions = []

rf_loaded = joblib.load('../xgb_time_aug_new_1229.pkl')
label_loaded = joblib.load('../xgb_time_aug_new_1229_labels.pkl')
dp_bandwidth=90

all=0

for tp, pp, dp in VALID_PARALLEL:
    device=['v100' for i in range(pp)]
    pp_bandwidth=[90 for i in range(pp-1)]
    
    for i, params in enumerate(combinations):
        # print(f"TP{tp} PP{pp} DP{dp} [{i+1}/{len(combinations)}] Running with params: {params}")
        all+=1
        if dp * params["MICRO_BATCH_SIZE"] > GLOBAL_BATCH_SIZE:
            #print(f"DP_SIZE {dp} * MICRO_BATCH_SIZE {params["MICRO_BATCH_SIZE"]} cannot be large than GLOBAL_BATCH_SIZE {GLOBAL_BATCH_SIZE}!")
            continue

        if params["MAX_POSITION_EMBEDDINGS"] < params["SEQ_LENGTH"]:
            #print(f"MAX_POSITION_EMBEDDINGS {params['MAX_POSITION_EMBEDDINGS']} cannot be less than SEQ_LENGTH {params['SEQ_LENGTH']}!")
            continue
        
        if params["NUM_QUERY_GROUPS"] < params["NUM_ATTENTION_HEADS"]:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} cannot be less than NUM_ATTENTION_HEADS {params['NUM_ATTENTION_HEADS']}!")
            continue
        
        if params["NUM_QUERY_GROUPS"] < tp:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} must be a multiple of tensor_parallel_size ({str(tp)})")
            continue
        
        if params["NUM_LAYERS"] < pp:
            #print(f"NUM_LAYERS {params['NUM_LAYERS']} cannot be less than pipeline_parallel_size ({str(pp)})")
            continue
        
        if params["NUM_ATTENTION_HEADS"] % tp != 0:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} must be a multiple of tensor_parallel_size ({str(tp)})")
            continue
        
        if params["NUM_ATTENTION_HEADS"] % params["NUM_QUERY_GROUPS"] != 0:
            #print(f"AssertionError: The number of attention heads {params["NUM_ATTENTION_HEADS"]} must be divisible by the number of GQA groups {params["NUM_QUERY_GROUPS"]}! when instantiating TEDotProductAttention")
            continue

        for granu in RECOMPUTE_GRANULARITY:
            if granu == 'full':
                RECOMPUTE_NUM_LAYERS = [
                    n for n in range(1, min(2, params["NUM_LAYERS"]//pp) + 1)
                ]  
                for method in RECOMPUTE_METHOD:
                    for re_nl in RECOMPUTE_NUM_LAYERS:
                        config = {
                            'global_batch_size': GLOBAL_BATCH_SIZE,
                            'micro_batch_size': params["MICRO_BATCH_SIZE"],
                            'DP_size':dp,
                            'TP_size':tp,
                            'PP_size':pp,
                            'sequence_length':params["SEQ_LENGTH"],
                            'hidden_state':HIDDEN_SIZE,
                            'dtype_bytes':2,#fp16
                            'num_layers':params["NUM_LAYERS"],
                            'ffn_hidden_state':FFN_HIDDEN_SIZE,
                            'vocab_size':VOCAB_SIZE,
                            'max_position_embedding': params["MAX_POSITION_EMBEDDINGS"],
                            'num_attention_heads': params["NUM_ATTENTION_HEADS"],
                            'num_query_groups': params["NUM_QUERY_GROUPS"], 
                            'recompute_granularity': granu+'-'+method+'-'+str(re_nl),
                        }
                        e2e_time = e2e_performance.amp_e2e_time(config, device, pp_bandwidth, dp_bandwidth, rf_loaded, label_loaded)
                        predictions.append((e2e_time,config))
            else:
                config = {
                    'global_batch_size': GLOBAL_BATCH_SIZE,
                    'micro_batch_size': params["MICRO_BATCH_SIZE"],
                    'DP_size':dp,
                    'TP_size':tp,
                    'PP_size':pp,
                    'sequence_length':params["SEQ_LENGTH"],
                    'hidden_state':HIDDEN_SIZE,
                    'dtype_bytes':2,#fp16
                    'num_layers':params["NUM_LAYERS"],
                    'ffn_hidden_state':FFN_HIDDEN_SIZE,
                    'vocab_size':VOCAB_SIZE,
                    'max_position_embedding': params["MAX_POSITION_EMBEDDINGS"],
                    'num_attention_heads': params["NUM_ATTENTION_HEADS"],
                    'num_query_groups': params["NUM_QUERY_GROUPS"], 
                    'recompute_granularity': granu,
                }
                e2e_time = e2e_performance.amp_e2e_time(config, device, pp_bandwidth, dp_bandwidth, rf_loaded, label_loaded)
                predictions.append((e2e_time,config))

predictions=sorted(predictions, key=lambda x: x[0])

print("all:",all)
print("length:",len(predictions))
print("top 10:",predictions[:10])

print("search time:", time.time()-start_time)