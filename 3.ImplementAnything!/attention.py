import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head

    def forward(self, Q, K, V, mask):
        # 1. Q*Kt MatMul
        scores = torch.matmul(Q, K.transpose(-1, -2))  # (batchsize, -, -) 형태이므로 -1,-2만 transpose

        # 2. Scale
        scaled_scores = scores / self.d_head ** 0.5

        # 3. Mask(opt.)
        scaled_scores = scaled_scores.masked_fill(mask, -1e9)

        # 4. Softmax
        prob = torch.softmax(scaled_scores, dim=-1)

        # 5. MatMul
        context = torch.matmul(prob, V)

        return context, prob