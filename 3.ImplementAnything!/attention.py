# Reference : https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
# Reference : https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

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

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, d_hidden, d_key = None, d_value = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        assert self.d_hidden%self.num_heads == 0, "d_hidden must be divisible by num_heads."
        self.d_head = self.d_hidden//self.num_heads

        # self.d_key = d_hidden if d_key is None else d_key
        # self.d_value = d_hidden if d_value is None else d_value

        self.w_query = nn.Parameter(torch.randn(self.d_hidden, self.num_heads * self.d_head))
        self.w_key = nn.Parameter(torch.randn(self.d_hidden, self.num_heads * self.d_head))
        self.w_value = nn.Parameter(torch.randn(self.d_hidden, self.num_heads * self.d_head))

        self.scaled_dot_attention = ScaledDotProductAttention(d_head = self.num_heads * self.d_head)

        self.last_linear = nn.Linear(self.num_heads * self.d_head, self.d_hidden) # nn.Linear ?

    def linear(self, A, x, b = None):
        """
        Shape :
            - x : (N, *, in_features) * ; additional dimensions
            - A : (out_features, in_features)
            - b : (out_features,)
            - Outputs : (N, *, out_feautures)

        return : x*AT + b
        """

        if b is None:
            b = torch.zeros(A.shape[0])

        return torch.matmul(x,A.transpose(-1,-2)) + b

    def forward(self, Q, K, V, mask):
        """
        Inputs
            - query : (batch size, target sequence length, embedding dimension)
            - key : (batch size, source sequence length, embedding dimension)
            - value : (batch size, source sequence length, embedding dimension)

        Outputs
            - outputs : (batch size, target sequence length, embedding dimension
            - weights : (batch size, target sequence length, source sequence length)
        """
        q = self.linear(A = self.w_query, x = Q, b = None)
        k = self.linear(A = self.w_key, x = K, b = None)
        v = self.linear(A = self.w_value, x = V, b = None)

        # mask = mask.unsqueeze(dim = 1).repeat(1,self.num_heads,1,1)

        context, prob = self.scaled_dot_attention(Q = q, K = k, V = v, mask = mask)

        output = self.last_linear(context)

        return output, prob