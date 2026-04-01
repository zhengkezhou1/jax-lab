import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


class GroupRMSNorm(nnx.Module):
    """Group RMS normalization.

    Splits the last dimension into `num_groups` groups and applies
    RMSNorm independently within each group.
    """

    def __init__(
        self,
        hidden_size: int,
        num_groups: int = 8,
        epsilon: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        self.epsilon = epsilon
        self.weight = nnx.Param(jnp.ones(shape=hidden_size, dtype=param_dtype))

        assert hidden_size % num_groups == 0, "hidden_size must be divisible by num_groups"

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        # reshape → square → sum → div → rsqrt → mul → weight_mul
        orig_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(*orig_shape[:-1], self.num_groups, self.group_size).astype(jnp.float32)
        variance = jnp.mean(lax.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * lax.rsqrt(variance + self.epsilon)
        hidden_states = hidden_states.astype(orig_dtype).reshape(orig_shape)
        return self.weight[...] * hidden_states
