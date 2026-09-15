import math
import torch

def torchAttention(q, k, v, causal=False):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        mask = torch.tril(torch.ones(d, d, device=q.device)).view(1, 1, d, d)       # lower triangle mask
        scores = scores.masked_fill(mask == 0, float('-inf'))           # Setting the valueus above to -inf, so that exp becomes 0
    
    probs = torch.softmax(scores, dim=-1)

    output = torch.matmul(probs, v)

    return output