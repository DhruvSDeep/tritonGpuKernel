# README

### Hardware:
Tested on a RTX 4050 GPU using torch 2.6.0+cu124 on windows with windows-triton ver. 3.8.0.post28

### Repo structure:
pytorchAttention.py: PyTorch baseline for attention.
tritonAttention.py: Triton kernel for fused attention.
harness.py: Correctness test script.
benchmark.py: Script to measure latency and memory.
autotune.py: Script to test different tile sizes.
WRITEUP.md: Project writeup.
benchmark_results.png: Plot of benchmark results.

Run the 4dotx.py for the part mentioned in 4.x of the task
WRITEUP.md holds the hand-written explanation
benchmark_results.png is the graph plot against naive and sdpf
