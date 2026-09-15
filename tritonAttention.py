import torch
import triton
import triton.language as trilang

def triAttention(Q, K, V, Out,           # q,k,v and output addresses
    stride_qb, stride_qh, stride_qm, stride_qd,        # b for batch, h for head
    stride_kb, stride_kh, stride_kn, stride_kd,        # m for query seq, n for kv seq
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim, sm_scale,
    BLOCK_M: trilang.constexpr, BLOCK_N: trilang.constexpr, BLOCK_DMODEL: trilang.constexpr, causal=False):

    batch_id = trilang.program_id(0)
    head_id = trilang.program_id(1)
    start_m = trilang.program_id(2)

    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh         # starting points
    o_offset = batch_id * stride_ob + head_id * stride_oh

    offs_m = start_m * BLOCK_M + trilang.arange(0, BLOCK_M)     # basically the tile we want to tile over
    offs_n = trilang.arange(0, BLOCK_N)
    offs_d = trilang.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd       
    #essentially gets the coordinates for the 2d tile, represented in 1d for triton
    

    q = trilang.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0) # load q into sram
    # by default looks for powers of 2. The mask takes care of those seqs which arent a power of 2
    # we load the whole sequence of q, and then do the calc by loading partial k,v in the for loop


    m_i = trilang.zeros([BLOCK_M], dtype=trilang.float32) - float("inf")        #running maximum
    l_i = trilang.zeros([BLOCK_M], dtype=trilang.float32)                       # running sigma e denom
    acc = trilang.zeros([BLOCK_M, BLOCK_DMODEL], dtype=trilang.float32)         # accumulator, basically running sum of unnormalised softmaxes
    #float32 for precision


    k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd   #ssame as what happened to q
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd


    if causal:
        n_blocks = n_blocks = -1 * (-1*((start_m * BLOCK_M) + BLOCK_M) // BLOCK_N)          # if its causal we only iterate over the blocks of the q until now, not whole sequence
    else:    
        n_blocks = -1 * (-1*seq_len // BLOCK_N)     #going over the qhole sequence

    for start_n in range(0, n_blocks):
        start_n_offset = start_n * BLOCK_N      # this is each block of k/v
        
        k = trilang.load(k_ptrs, mask=(start_n_offset + offs_n)[None, :] < seq_len, other=0.0)  #loading k,v 
        v = trilang.load(v_ptrs, mask=(start_n_offset + offs_n)[:, None] < seq_len, other=0.0)

        qk = trilang.zeros([BLOCK_M, BLOCK_N], dtype=trilang.float32)         # dot prod the q and the k block
        qk += trilang.dot(q, k)
        qk = qk * sm_scale

        if causal:
            causal_mask = offs_m[:, None] >= (start_n_offset + offs_n)[None, :]     #create the lower triangle mask
            qk = trilang.where(causal_mask, qk, float("-inf"))          #set vals to -inf, so that exp takes them to 0

        m_ij = trilang.max(qk, 1)            # current max
        m_new = trilang.maximum(m_i, m_ij)   # global max
        
        alpha = trilang.exp(m_i - m_new)            # the rescale factor
        beta = trilang.exp(qk - m_new[:, None])     # the current block's subtracted exponents, which is softmax numerator
        
        l_i = l_i * alpha + trilang.sum(beta, 1)        # updating the running denom as rescaled past denom + current denom
        acc = acc * alpha[:, None]                      # rescale the accumulated value also


        p = beta.to(v.dtype) # typecast beta, the numerator, to values type. Here its mainly to change precisions


        acc += trilang.dot(p, v)    # Add the dot prod of p and v
        # to note that in the end we will get attention by dividing final acc with final l_i

        m_i = m_new     

        k_ptrs += BLOCK_N * stride_kn           #increment the k and v pointers for next iter
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]            # final attention score

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od     #output offsetting
    trilang.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)     #final storing vals
