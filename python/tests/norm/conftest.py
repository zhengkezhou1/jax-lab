import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

_test_norm_dir = Path(__file__).resolve().parent

# Add model/norm/ so tests can import group_rms_norm_jax.
sys.path.insert(0, str(_test_norm_dir.parent.parent / "model" / "norm"))
# Add tests/norm/ so tests can import conftest helpers.
sys.path.insert(0, str(_test_norm_dir))

from group_rms_norm import GroupRMSNorm  # noqa: E402

# --- PyTorch reference implementation ---


class BailingMoeV2_5GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, param_dtype, eps=1e-6):
        """BailingMoeV2_5RMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=param_dtype))
        self.group_norm_size = group_norm_size
        if hidden_size % group_norm_size != 0:
            raise ValueError("hidden_size must be divisible by group_norm_size")
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


def make_jax_model(hidden_size, num_groups, epsilon, weight=None, param_dtype=jnp.float32):
    model = GroupRMSNorm(hidden_size, num_groups=num_groups, epsilon=epsilon, param_dtype=param_dtype)
    if weight is not None:
        model.weight[...] = jnp.array(weight, dtype=param_dtype)
    return model


def make_torch_model(hidden_size, num_groups, epsilon, weight=None, param_dtype=torch.float32):
    model = BailingMoeV2_5GroupRMSNorm(hidden_size, num_groups, param_dtype=param_dtype, eps=epsilon)
    if weight is not None:
        model.weight = torch.nn.Parameter(torch.tensor(weight, dtype=param_dtype))
    return model


def run_jax(model, input_np, dtype=jnp.float32):
    return np.array(model(jnp.array(input_np, dtype=dtype)))


def run_torch(model, input_np, dtype=torch.float32):
    return model(torch.tensor(input_np, dtype=dtype)).detach().float().numpy()


def numpy_group_rmsnorm_fp64(input_data, weight, num_groups):
    """fp64 ground truth reference implementation."""
    x = input_data.astype(np.float64)
    w = weight.astype(np.float64)
    orig_shape = x.shape
    group_size = orig_shape[-1] // num_groups
    x = x.reshape(*orig_shape[:-1], num_groups, group_size)
    variance = np.mean(x**2, axis=-1, keepdims=True)
    x = x / np.sqrt(variance + 1e-6)
    x = x.reshape(orig_shape)
    return w * x
