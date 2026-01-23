# csv_writer.py
import os, csv, io, subprocess, re
import torch
import torch.distributed as dist

CSV_FILE = 'result_recompute.csv'
_row_dict = {}

def write_csv_column(name, value):
    """写某一列数据"""
    global _row_dict
    try:
        v = float(value)
        _row_dict[name] = f"{v:g}"
    except (TypeError, ValueError):
        _row_dict[name] = str(value)

@torch.no_grad()
def write_csv_newline():
    """
    使用 all_gather_object 把各 rank 的 dict 一次性收齐，
    由 rank0 统一写 CSV。
    """
    world_size = dist.get_world_size()
    rank       = dist.get_rank()

    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, _row_dict)

    if rank == 0:
        all_keys = {k for d in gathered_list for k in d}

        first_cols = ['TP size']
        last_cols = ['Peak GPU memory', 'TFLOP/s/GPU', 'elapsed time per iteration']

        rest_cols = sorted(all_keys - set(first_cols)- set(last_cols))

        final_header = first_cols + rest_cols + last_cols

        first_write = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_header)
            if first_write:
                writer.writeheader()
            for row in gathered_list:   # 每个 rank 一行
                writer.writerow(row)

    dist.barrier()
    _row_dict.clear()

def gpu_memory_used(gpu_id=0):
    cmd = ["nvidia-smi",
           "--query-gpu=memory.used",
           "--format=csv,noheader,nounits",
           "-i", str(gpu_id)]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out)