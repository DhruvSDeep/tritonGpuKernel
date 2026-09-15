import torch
import math
import os, tempfile, uuid
os.environ["TRITON_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), f"triton_cache_{uuid.uuid4().hex}")
import triton
import triton.language as trilang

from pytorchAttention import attention as naive
from tritonAttention import triAttention_kickstart as cunning



def make_inputs(batch_size, num_heads, seq_len, head_dim, device, magnitude):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16) * magnitude
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16) * magnitude
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    return q, k, v

def run_correctness_harness():
    torch.manual_seed(4000)
    device = torch.device("cuda")

    shape_cases = [
        (1, 2, 64, 64, "Small"),
        (2, 4, 256, 64, "Standard"),
        (1, 1, 200, 32, "Non power of two seq len"),
        (2, 8, 512, 256, "Larger head dim"),
        (1, 2, 2048, 64, "Longer seq len"),
    ]

    input_variants = [
        (1.0, "random"),
        (50.0, "Adversarial large magnitude"),
    ]

    causal_variants = [False, True]

    rtol, atol = 1e-2, 1e-2

    passed_tests = 0
    total_tests = 0

    for batch_size, num_heads, seq_len, head_dim, desc in shape_cases:
        for magnitude, input_desc in input_variants:
            for causal in causal_variants:
                print()
                print()
                total_tests += 1
                print(f"Test {total_tests}: {desc}, {input_desc}, causal={causal}")
                print(f"Shape B:{batch_size}, H:{num_heads}, N:{seq_len}, d:{head_dim}")

                q, k, v = make_inputs(batch_size, num_heads, seq_len, head_dim, device, magnitude)

                ref_out = naive(q, k, v, causal)
                try:
                    tri_out = cunning(q, k, v, causal)
                except Exception as e:
                    print(f"FAIL: Triton kernel threw an exception: {e}")
                    continue

                if torch.isnan(tri_out).any() or torch.isinf(tri_out).any():
                    print("FAIL: Triton output contains NaNs or Infs")
                    continue
                if torch.isnan(ref_out).any() or torch.isinf(ref_out).any():
                    print("PASS: PyTorch output contains NaNs or Infs")
                    passed_tests += 1
                    continue

                match = torch.allclose(ref_out, tri_out, rtol=rtol, atol=atol)
                if match:
                    print("PASS")
                    passed_tests += 1
                else:
                    max_diff = (ref_out - tri_out).abs().max().item()
                    print(f"FAIL: Outputs do not match. Max absolute difference: {max_diff:.6f}")

    print("Harness Complete")
    print(f"Passed {passed_tests}/{total_tests} test suites successfully.")

run_correctness_harness()