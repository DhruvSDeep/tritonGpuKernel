import math
import torch

def attention(q, k, v, causal=False):

    B, H, N, d = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        mask = torch.tril(torch.ones(N, N, device=q.device)).view(1, 1, N, N)       # lower triangle mask
        scores = scores.masked_fill(mask == 0, float('-inf'))           # Setting the valueus above to -inf, so that exp becomes 0
    
    probs = torch.softmax(scores, dim=-1)

    output = torch.matmul(probs, v)

    return output