import torch
from torch import nn

class BailingMoeV2_5GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        """
        BailingMoeV2_5RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert hidden_size % group_norm_size == 0, "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)