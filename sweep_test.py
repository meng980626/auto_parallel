import itertools
from itertools import product
from collections import Counter
    

TOTAL_GPU = 4
VALID_PARALLEL = []

for tp in range(1, TOTAL_GPU + 1):
    if (tp & (tp - 1)) != 0:
        continue
    # for pp in range(1, TOTAL_GPU + 1):   
    #     if (pp & (pp - 1)) != 0:
    #         continue
    #     for dp in range(1, TOTAL_GPU + 1):   
    #         if (dp & (dp - 1)) != 0:
    #             continue
    #         if tp * pp * dp <= TOTAL_GPU:   
    #             VALID_PARALLEL.append((tp, pp, dp))
    VALID_PARALLEL.append((tp, 1, 1))

GLOBAL_BATCH_SIZE = 32
FFN_HIDDEN_SIZE = 11008
HIDDEN_SIZE = 4096
CP_SIZE = 1
DTYPE = ['fp16']
# if args.device == "a100":
#     DTYPE = ['fp16','bf16']

recompute_method = []

param_grid = {
    "NUM_LAYERS": [1,2,4,8,16,32],
    "SEQ_LENGTH": [1024,2048,4096],
    "MICRO_BATCH_SIZE": [x for x in range(1, GLOBAL_BATCH_SIZE+1) if (x & (x - 1)) == 0],
    "MAX_POSITION_EMBEDDINGS": [2048,4096,8192],
    "NUM_ATTENTION_HEADS": [1,2,4,8,16,32],
    "NUM_QUERY_GROUPS": [1,2,4,8,16,32],
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
num_experiments = 0

min_experiment_dict = set()
num_layers_experiment_count = Counter()
seq_length_experiment_count = Counter()
micro_bs_experiment_count = Counter()

for tp, pp, dp in VALID_PARALLEL:
    for i, params in enumerate(combinations):
        # print(f"TP{tp} PP{pp} DP{dp} [{i+1}/{len(combinations)}] Running with params: {params}")
        if params["MAX_POSITION_EMBEDDINGS"] < params["SEQ_LENGTH"]:
            #print(f"MAX_POSITION_EMBEDDINGS {params['MAX_POSITION_EMBEDDINGS']} cannot be less than SEQ_LENGTH {params['SEQ_LENGTH']}!")
            continue
        
        if params["NUM_QUERY_GROUPS"] < params["NUM_ATTENTION_HEADS"]:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} cannot be less than NUM_ATTENTION_HEADS {params['NUM_ATTENTION_HEADS']}!")
            continue
        
        if params["NUM_QUERY_GROUPS"] % tp != 0:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} must be a multiple of tensor_parallel_size ({str(tp)})")
            continue
        
        if params["NUM_LAYERS"] < pp:
            #print(f"NUM_LAYERS {params['NUM_LAYERS']} cannot be less than pipeline_parallel_size ({str(pp)})")
            continue
        
        if params["NUM_ATTENTION_HEADS"] % tp != 0:
            #print(f"NUM_ATTENTION_HEADS {params['NUM_ATTENTION_HEADS']} must be a multiple of tensor_parallel_size ({str(tp)})")
            continue
        
        if params["NUM_ATTENTION_HEADS"] % params["NUM_QUERY_GROUPS"] != 0:
            #print(f"AssertionError: The number of attention heads {params["NUM_ATTENTION_HEADS"]} must be divisible by the number of GQA groups {params["NUM_QUERY_GROUPS"]}! when instantiating TEDotProductAttention")
            continue
        
        key_num_layers = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["MICRO_BATCH_SIZE"], params["SEQ_LENGTH"])

        num_layers_experiment_count[key_num_layers]+=1

        key_seq_length = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["MICRO_BATCH_SIZE"])

        seq_length_experiment_count[key_seq_length]+=1

        key_micro_bs = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["SEQ_LENGTH"])

        micro_bs_experiment_count[key_micro_bs]+=1
        
        if num_layers_experiment_count[key_num_layers]<=2 and seq_length_experiment_count[key_seq_length]<=2 and micro_bs_experiment_count[key_micro_bs] <=2:
            key = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["MICRO_BATCH_SIZE"], params["SEQ_LENGTH"]) 
            min_experiment_dict.add(key)

print(len(min_experiment_dict))
recompute_num=0

RECOMPUTE_GRANULARITY = ['full','selective','null']
RECOMPUTE_METHOD = ['uniform','block']

for tp, pp, _, _, _, _, num_layers, _, _ in min_experiment_dict:
    for granu in RECOMPUTE_GRANULARITY:
        if granu == 'null':
            recompute_num+=1
        elif granu == 'selective':
            recompute_num+=1
        else: # granu == 'full'
            #暂时假定NUM_LAYERS能被pp整除，实际上有可能会不均匀切分
            RECOMPUTE_NUM_LAYERS = [
                n for n in range(1, num_layers // pp + 1)
            ]  
            for method in RECOMPUTE_METHOD:
                for re_nl in RECOMPUTE_NUM_LAYERS:
                    if tp > 1: #可以开启DISTRIBUTE_SAVED_ACTIVATIONS
                        recompute_num+=2
                    else:
                        recompute_num+=1

print(recompute_num)



