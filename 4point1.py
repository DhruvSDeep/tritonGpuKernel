import torch
device = torch.device("cuda")
from pytorchAttention import attention


B, H, d = 1, 8, 256

seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768] # bigger seq lens to force OOM

for N in seq_lengths:
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        q = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, d, device=device, dtype=torch.float16)

       
        out = attention(q, k, v, True)      # actual run

        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"seq length: {N} and peak memory: {peak_mem_mb:.2f} MB")

        del q, k, v, out        #clean in case

    except torch.cuda.OutOfMemoryError:
        print(f"OOM at seq length of {N}")
        break
