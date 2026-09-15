# FLASH ATTENTION
<small>*p.s. this is completely human-written, based on my understanding of what I've built and some research on specific topics (like arithmetic intensity.)*</small>

### Hardware:
Tested on a RTX 4050 GPU using torch 2.6.0+cu124 on windows with windows-triton ver. 3.8.0.post28

### What is basic attention?
The regular attention is softmax( (Q . Kt)/root(d) ) . V
Where softmax is e^xi / sigma(e^xj) for all xj.
Basically this means that a single term is also dependent on the sum of all the terms.
Also, the exponents in this case could very easily blow up, as we are using float16, so we subtract the maximum value of the row from each value before taking the exponent.

### Pytorch approach:
When Q . Kt is calculated, there is an nxn matrix which now must be stored on the High Bandwidth Memory (HBM). This means the memory complexity is now n^2.
Ofcourse, this HBM is slower to read than SRAM, and over the course of the operation, pytorch will need to load this data to and from the HBM multiple times.

### Our approach:
We dont need to calculate the entire nxn in one go, instead we will Load batches into SRAM, one at a time, and update the outcome array as we go.
This happens by loading the query, then loading and calculating the key and value part of it in batches.

### Hurdle:
Softmax of 1 term requires knowing the sum, and the maximum value of the entire row.

### How we get around it:
We calculate the value of the numerator using the max of each block. Then, if the next block has a higher maximum, we rescale the previous outputs with e^(m_old- m_new). In this way we calculate the entire attention's numerator. During this, we are also calculating and sequentially rescaling the denominator summation. In the end, we just divide the numerator by the denominator.

This part is kind of finnicky, as forgetting to rescale some part of it, or some other small mistake, could easily produce plausible but incorrect results, so I tested it against the standard pytorch for accuracy, and it matches.

### HBM Bandwidth vs. FLOPs:
There is a term, Arithmetic Intensity (I) = Compute flops / bytes transferred to/from hbm
This value of I is bounded by the hardware's peak flops/s
If we look at naive torch:
-initial mat mul takes 2N^2d flops, and also writes 2N^2 bytes to hbm to store the nxn.
-the intermediate compute in calculating softmax takes 4N^2 flops, and again 2N^2 bytes to hbm to read and 2N^2 to put output probs back.
-The last calculation which is multiplying probs with V takes another 2N^2d flops, and ofcourse needs to read the probs back from hbm, taking another 2N^2 bytes.

Overall, we get I as somewhere around d/2 FLOPs/byte.
Assuming something like d = 64, we get the naive I as 32.
A google search tells me my 4050 is capable of 375 FLOPs/byte with fp16.
So, it is evident that the HBM bandwidth is the bottleneck slowing down pytorch's naive attention.

### Adversarial Inputs:
We test with much higher inputs than torch randn (50x scaled), mostly to check how it will handle higher numbers. Because we have subtract each term by the max term before we exponentiate, the exponent ends up being negative, so the overall value doesn't start to overflow upwards.

### Causal masking and why:
In GPT style transformers, there isn't a bidirectional flow of data. So, each new word, can only see the words/tokens which come before it. You could do the calculation, then set all the later values to be discarded. So, we apply a triangular mask, and set the values we don't want to -inf, so that the exponentiation step takes them to zero.
However, while coding the gpu kernel ourselves in triton, we have more autonomy. We can just choose not to calculate the scores for those batches of k/v beyond the current q. There might be a few members of a batch left behind, which we would then calculate and throw out like before, but it is still a huge improvement.

### Benchmark:
benchmark_results.png is a plot showing the performance of naive pytorch, out triton, and scaled dot prod attention.
Clearly, for the naive torch, both memory and latency appear O(n^2), whereas the other 2 sit much lower down, near linear.

### AutoScaling:

| BLOCK_M | BLOCK_N | num_warps | Latency (ms) |
|---|---|---|---|
remark: 4dot7.py:18:0: 602 instructions in function
| 64      | 32      | 4         | 0.878 ms     |
remark: 4dot7.py:18:0: 866 instructions in function
| 64      | 64      | 4         | 0.884 ms     |
remark: 4dot7.py:18:0: 964 instructions in function
| 128     | 32      | 4         | 1.085 ms     |
remark: 4dot7.py:18:0: 1398 instructions in function
| 128     | 64      | 4         | 1.029 ms     |
remark: 4dot7.py:18:0: 1288 instructions in function
| 128     | 128     | 8         | 1.066 ms     |

Here is the output. We can see, the smaller m and n blocks perform better. This is because the larger blocks probably didn't fit on sram as efficiently packed as the smaller ones.
Also, we can see larger blocks having more instructions in function, which could mean the compiler had to spend more time reorganizing them in memory, to make them tile better.

### Why worse than cuBLAS?
CuBLAS is much more organised at the lower level than triton is. It uses strategies like asynchronous memory, by bypassing internal registers, highly sequenced progressions.
However, naive cuBLAS used by pytorch still has the limitations discussed earlier.
However, SDPF with CuBLAAS is a much closer competition.
As you can see in the png, my flash attention does infact beat out cuBLAAS on plain memory, while loses to it on speed.
CuBLAAS has more generality than my custom kernel (naturally,) and so the block sizes it might be using might be bigger. This probably leads to more memory usage than my triton, but translates to more efficient use, in terms of latency.
cuBLAAS also comes pre-built and constantly maintained by professionals, along with having the backprop support, which I couldn't find the time to implement (sorry, writing this took too long.) 

