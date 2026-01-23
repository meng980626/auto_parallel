import torch
import torch.distributed as dist
import time
import argparse
import numpy as np
import os
from datetime import datetime

def init_distributed(backend='nccl'):
    """初始化分布式环境"""
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()

def warmup(src_rank, dst_rank, size_bytes=1000000):
    """预热通信"""
    if src_rank == dst_rank:
        return
    
    rank = dist.get_rank()
    num_elements = size_bytes // 4
    warmup_data = torch.randn(num_elements, device='cuda')
    
    if rank == src_rank:
        for _ in range(5):
            dist.send(warmup_data, dst=dst_rank)
            # 等待接收方确认
            dist.recv(torch.zeros(1, device='cuda'), src=dst_rank)
    elif rank == dst_rank:
        for _ in range(5):
            dist.recv(warmup_data, src=src_rank)
            # 发送确认
            dist.send(torch.ones(1, device='cuda'), dst=src_rank)
    
    torch.cuda.synchronize()

def test_p2p_bandwidth(src_rank, dst_rank, size_bytes, num_iterations=10):
    """
    测试两个GPU之间的P2P带宽
    
    Args:
        src_rank: 源GPU的rank
        dst_rank: 目标GPU的rank
        size_bytes: 传输数据大小（字节）
        num_iterations: 测试迭代次数
    """
    rank = dist.get_rank()
    
    # 检查是否有效通信对
    if src_rank == dst_rank:
        if rank == 0:
            print(f"警告: 跳过自通信测试 (rank {src_rank} -> rank {dst_rank})")
        return None
    
    if rank not in [src_rank, dst_rank]:
        # 非参与进程进行同步
        dist.barrier()
        return None
    
    # 创建测试数据
    num_elements = size_bytes // 4  # 假设float32
    data = torch.randn(num_elements, device='cuda')
    
    # 预热
    warmup(src_rank, dst_rank, size_bytes)
    torch.cuda.synchronize()
    
    # 测试单向带宽
    if rank == src_rank:
        times = []
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            dist.send(data, dst=dst_rank)
            torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
            
            # 等待接收方确认
            dist.recv(torch.zeros(1, device='cuda'), src=dst_rank)
        
        # 使用去掉最大值和最小值后的平均值
        if len(times) > 4:
            times_sorted = sorted(times[1:])  # 忽略第一次
            times_filtered = times_sorted[1:-1]  # 去掉最小和最大值
            avg_time = np.mean(times_filtered)
        else:
            avg_time = np.mean(times[1:])
            
        bandwidth = (size_bytes / avg_time) / (1024**3)  # GB/s
        return bandwidth
    
    elif rank == dst_rank:
        for i in range(num_iterations):
            dist.recv(data, src=src_rank)
            torch.cuda.synchronize()
            # 发送确认
            dist.send(torch.ones(1, device='cuda'), dst=src_rank)
        
        return None

def test_p2p_pingpong(src_rank, dst_rank, size_bytes, num_iterations=10):
    """测试Ping-Pong延迟和带宽"""
    if src_rank == dst_rank:
        return None
    
    rank = dist.get_rank()
    
    if rank not in [src_rank, dst_rank]:
        dist.barrier()
        return None
    
    num_elements = size_bytes // 4
    data = torch.randn(num_elements, device='cuda')
    
    warmup(src_rank, dst_rank, size_bytes)
    torch.cuda.synchronize()
    
    if rank == src_rank:
        times = []
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            dist.send(data, dst=dst_rank)
            dist.recv(data, src=dst_rank)
            torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        # 计算平均往返时间
        if len(times) > 4:
            times_sorted = sorted(times[1:])
            times_filtered = times_sorted[1:-1]
            avg_time = np.mean(times_filtered)
        else:
            avg_time = np.mean(times[1:])
        
        # Ping-Pong带宽（双向）
        bandwidth = (2 * size_bytes / avg_time) / (1024**3)  # GB/s
        latency = avg_time * 1000 / 2  # 单向延迟，单位ms
        
        return bandwidth, latency
    
    elif rank == dst_rank:
        for i in range(num_iterations):
            dist.recv(data, src=src_rank)
            dist.send(data, dst=src_rank)
            torch.cuda.synchronize()
        
        return None

def test_allgather_bandwidth(size_bytes, num_iterations=5):
    """测试AllGather集体通信带宽"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    num_elements = size_bytes // 4
    local_data = torch.randn(num_elements, device='cuda')
    gathered_data = [torch.zeros_like(local_data) for _ in range(world_size)]
    
    # 预热
    for _ in range(3):
        dist.all_gather(gathered_data, local_data)
    torch.cuda.synchronize()
    
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        dist.all_gather(gathered_data, local_data)
        torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
    
    # 计算带宽
    if len(times) > 4:
        times_sorted = sorted(times[1:])
        times_filtered = times_sorted[1:-1]
        avg_time = np.mean(times_filtered)
    else:
        avg_time = np.mean(times[1:])
    
    # AllGather的总数据量：(world_size-1) * size_bytes
    total_bytes = (world_size - 1) * size_bytes
    bandwidth = (total_bytes / avg_time) / (1024**3)  # GB/s
    
    return bandwidth

def test_all_pairs_bandwidth(world_size, size_bytes=256*1024*1024, num_iterations=5):
    """测试所有GPU对之间的带宽"""
    rank = dist.get_rank()
    all_results = {}
    
    # 只在rank 0收集结果
    if rank == 0:
        print(f"\n{'='*80}")
        print("P2P带宽测试结果 (单向)")
        print(f"{'='*80}")
        print(f"{'源GPU':<8} {'目标GPU':<8} {'带宽(GB/s)':<12} {'延迟(ms)':<12} {'同节点':<8}")
        print(f"{'-'*80}")
    
    # 测试所有可能的对
    for src in range(world_size):
        for dst in range(world_size):
            if src >= dst:  # 避免重复测试
                continue
            
            # 同步
            dist.barrier()
            
            # 测试Ping-Pong（可以得到延迟和带宽）
            result = test_p2p_pingpong(src, dst, size_bytes, num_iterations)
            
            # 收集结果
            if rank == src:
                if result:
                    bandwidth, latency = result
                    # 判断是否在同一节点
                    same_node = False
                    try:
                        # 尝试通过CUDA设备属性判断
                        src_device = torch.cuda.get_device_properties(src % torch.cuda.device_count())
                        dst_device = torch.cuda.get_device_properties(dst % torch.cuda.device_count())
                        same_node = (src_device.name == dst_device.name)
                    except:
                        same_node = "未知"
                    
                    # 发送结果到rank 0
                    result_tensor = torch.tensor([bandwidth, latency], device='cuda')
                    dist.send(result_tensor, dst=0)
            
            elif rank == 0 and src != 0:
                # 接收结果
                result_tensor = torch.zeros(2, device='cuda')
                dist.recv(result_tensor, src=src)
                bandwidth, latency = result_tensor.cpu().numpy()
                
                all_results[(src, dst)] = {
                    'bandwidth': bandwidth,
                    'latency': latency,
                    'same_node': same_node if same_node != "未知" else "未知"
                }
            
            # rank 0 测试自己的对
            if rank == 0 and src == 0:
                if result:
                    bandwidth, latency = result
                    # 判断是否在同一节点
                    same_node = False
                    try:
                        src_device = torch.cuda.get_device_properties(0)
                        dst_device = torch.cuda.get_device_properties(dst % torch.cuda.device_count())
                        same_node = (src_device.name == dst_device.name)
                    except:
                        same_node = "未知"
                    
                    all_results[(0, dst)] = {
                        'bandwidth': bandwidth,
                        'latency': latency,
                        'same_node': same_node if same_node != "未知" else "未知"
                    }
    
    # 在rank 0打印结果
    if rank == 0:
        # 按带宽排序
        sorted_pairs = sorted(all_results.items(), 
                            key=lambda x: x[1]['bandwidth'], 
                            reverse=True)
        
        for (src, dst), metrics in sorted_pairs:
            print(f"{src:<8} {dst:<8} {metrics['bandwidth']:<12.2f} "
                  f"{metrics['latency']:<12.3f} {str(metrics['same_node']):<8}")
        
        print(f"{'='*80}")
        
        # 统计信息
        if all_results:
            bandwidths = [m['bandwidth'] for m in all_results.values()]
            latencies = [m['latency'] for m in all_results.values()]
            
            print(f"\n统计信息:")
            print(f"  平均带宽: {np.mean(bandwidths):.2f} ± {np.std(bandwidths):.2f} GB/s")
            print(f"  平均延迟: {np.mean(latencies):.3f} ± {np.std(latencies):.3f} ms")
            print(f"  最高带宽: {max(bandwidths):.2f} GB/s (GPU {sorted_pairs[0][0][0]} -> GPU {sorted_pairs[0][0][1]})")
            print(f"  最低延迟: {min(latencies):.3f} ms")
    
    return all_results if rank == 0 else None

def main():
    parser = argparse.ArgumentParser(description='GPU P2P带宽测试工具')
    parser.add_argument('--size-mb', type=int, default=256,
                       help='测试数据大小 (MB)，默认: 256')
    parser.add_argument('--iterations', type=int, default=10,
                       help='每个测试的迭代次数，默认: 10')
    parser.add_argument('--test-mode', type=str, default='p2p',
                       choices=['p2p', 'pingpong', 'allgather', 'all-pairs'],
                       help='测试模式: p2p(单向), pingpong(双向), allgather, all-pairs(所有对)')
    parser.add_argument('--src-rank', type=int, default=0,
                       help='源GPU rank，默认: 0')
    parser.add_argument('--dst-rank', type=int, default=1,
                       help='目标GPU rank，默认: 1')
    parser.add_argument('--output', type=str, default=None,
                       help='结果输出文件 (JSON格式)')
    
    args = parser.parse_args()
    
    # 初始化分布式环境
    rank, world_size = init_distributed()
    
    if rank == 0:
        print(f"PyTorch分布式GPU带宽测试工具")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"World size: {world_size}")
        print(f"本地GPU数量: {torch.cuda.device_count()}")
        print(f"测试模式: {args.test_mode}")
        print(f"数据大小: {args.size_mb} MB")
        print(f"迭代次数: {args.iterations}")
    
    size_bytes = args.size_mb * 1024 * 1024
    
    # 根据测试模式执行不同的测试
    if args.test_mode == 'all-pairs':
        results = test_all_pairs_bandwidth(world_size, size_bytes, args.iterations)
        
        # 保存结果到文件
        if rank == 0 and args.output:
            import json
            # 将numpy类型转换为Python原生类型
            def convert_numpy(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                raise TypeError
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)
            print(f"\n结果已保存到: {args.output}")
    
    elif args.test_mode == 'pingpong':
        if args.src_rank == args.dst_rank:
            if rank == 0:
                print("错误: 源和目标rank不能相同")
            dist.destroy_process_group()
            return
        
        result = test_p2p_pingpong(args.src_rank, args.dst_rank, 
                                  size_bytes, args.iterations)
        
        if rank == args.src_rank and result:
            bandwidth, latency = result
            print(f"\nPing-Pong测试结果 (GPU {args.src_rank} <-> GPU {args.dst_rank}):")
            print(f"  双向带宽: {bandwidth:.2f} GB/s")
            print(f"  单向延迟: {latency:.3f} ms")
    
    elif args.test_mode == 'allgather':
        bandwidth = test_allgather_bandwidth(size_bytes, args.iterations)
        if rank == 0:
            print(f"\nAllGather测试结果:")
            print(f"  有效带宽: {bandwidth:.2f} GB/s")
    
    else:  # p2p模式
        if args.src_rank == args.dst_rank:
            if rank == 0:
                print("错误: 源和目标rank不能相同")
            dist.destroy_process_group()
            return
        
        bandwidth = test_p2p_bandwidth(args.src_rank, args.dst_rank, 
                                      size_bytes, args.iterations)
        
        if rank == args.src_rank and bandwidth:
            print(f"\n单向P2P测试结果 (GPU {args.src_rank} -> GPU {args.dst_rank}):")
            print(f"  带宽: {bandwidth:.2f} GB/s")
    
    # 同步并清理
    dist.barrier()
    if rank == 0:
        print(f"\n测试完成!")
    dist.destroy_process_group()

if __name__ == "__main__":
    # 设置环境变量（如果未设置）
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    
    main()