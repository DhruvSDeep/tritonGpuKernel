import math
import torch

def attention(q, k, v):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    
    probs = torch.softmax(scores, dim=-1)

    output = torch.matmul(probs, v)

    return output