import torch
import triton
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorchAttention import attention as naive
from tritonAttention import triAttention_kickstart as cunning

def run_benchmarks():
    device = torch.device("cuda")
    B, H, d = 2, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    

    mem_naive, mem_sdpa, mem_triton = [], [], []    # storage for plot data1
    time_naive, time_sdpa, time_triton = [], [], []


    for N in seq_lengths:
        print(f"current seq len: {N}")

        # memory benchmark
        for name, func in [("Naive", naive), ("SDPA", F.scaled_dot_product_attention), ("Triton", cunning)]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
            q = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
            k = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
            v = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
            
            try:
                if name == "SDPA":
                    _ = func(q, k, v, is_causal=True)
                else:
                    _ = func(q, k, v, causal=True)
                    
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # In MB
            except torch.cuda.OutOfMemoryError:
                peak_mem = float('inf') # mark OOM
            
            if name == "Naive": mem_naive.append(peak_mem)
            elif name == "SDPA": mem_sdpa.append(peak_mem)
            elif name == "Triton": mem_triton.append(peak_mem)

            # cleaning
            del q, k, v
            torch.cuda.empty_cache()

        # latency bBenchmark
        q = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, d, device=device, dtype=torch.float16)

        # naive latency
        if mem_naive[-1] != float('inf'):
            time_naive.append(triton.testing.do_bench(lambda: naive(q, k, v, causal=True)))
        else:
            time_naive.append(None)
            
        # SDPA Latency
        time_sdpa.append(triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)))
        
        # Triton Latency
        time_triton.append(triton.testing.do_bench(lambda: cunning(q, k, v, causal=True)))

    return seq_lengths, mem_naive, mem_sdpa, mem_triton, time_naive, time_sdpa, time_triton

def plot_results(seq_lengths, mem_naive, mem_sdpa, mem_triton, time_naive, time_sdpa, time_triton):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # plotting memory
    ax1.plot(seq_lengths, mem_naive, label='Naive PyTorch (O(N^2))', marker='o', linestyle='--')
    ax1.plot(seq_lengths, mem_sdpa, label='PyTorch SDPA', marker='s')
    ax1.plot(seq_lengths, mem_triton, label='Custom Triton (O(N))', marker='^')
    ax1.set_title('Peak Memory vs Sequence Length')
    ax1.set_xlabel('Sequence Length (N)')
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.legend()
    ax1.grid(True)

    # now plotting teh latency
    ax2.plot(seq_lengths, time_naive, label='Naive PyTorch', marker='o', linestyle='--')
    ax2.plot(seq_lengths, time_sdpa, label='PyTorch SDPA', marker='s')
    ax2.plot(seq_lengths, time_triton, label='Custom Triton', marker='^')
    ax2.set_title('Wall-Clock Latency vs Sequence Length')
    ax2.set_xlabel('Sequence Length (N)')
    ax2.set_ylabel('Latency (ms)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()


results = run_benchmarks()
plot_results(*results)