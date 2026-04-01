import jax.numpy as jnp
import numpy as np
import pytest
import torch
from conftest import make_jax_model, make_torch_model, numpy_group_rmsnorm_fp64, run_jax, run_torch

HIDDEN_SIZE = 8192
NUM_GROUPS = 8
GROUP_SIZE = HIDDEN_SIZE // NUM_GROUPS  # 1024
EPSILON = 1e-6

# Machine epsilon
_eps_fp32 = float(jnp.finfo(jnp.float32).eps)  # 2^-23 ≈ 1.19e-7
_eps_bf16 = float(jnp.finfo(jnp.bfloat16).eps)  # 2^-7  ≈ 7.81e-3

# GroupRMSNorm 计算图：
# square(1 mul) → sum(GROUP_SIZE adds) → div(1) → rsqrt(1) → mul(1) → weight_mul(1)
# 每个框架累积误差：~(GROUP_SIZE + 5) · ε
# 跨框架 (JAX vs PyTorch): 2× → 2·(GROUP_SIZE + 5) · ε
_n_ops = GROUP_SIZE + 5

# fp32: 全程 fp32 计算
FP32_ATOL = 2 * _n_ops * _eps_fp32  # ≈ 5.0e-6

# bf16: fp32 normalize → cast bf16 → weight mul (bf16)
# fp32 跨框架误差 (~5e-6) << bf16 ULP (~7.8e-3)，通常舍入到同一 bf16 值
# 理论最坏情况：跨越 bf16 舍入边界 → 2 · ε_bf16
BF16_ATOL = 2 * _eps_bf16  # ≈ 1.56e-2


def _print_diff(label, jax_out, torch_out, atol):
    abs_diff = np.abs(jax_out - torch_out)
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  max  abs diff: {abs_diff.max():.6e}")
    print(f"  mean abs diff: {abs_diff.mean():.6e}")
    print(f"  theoretical atol: {atol:.6e}")
    print(f"{'=' * 50}")


@pytest.mark.parametrize(
    ("jax_dtype", "torch_dtype", "atol", "label"),
    [
        (jnp.float32, torch.float32, FP32_ATOL, "fp32"),
        (jnp.bfloat16, torch.bfloat16, BF16_ATOL, "bf16"),
    ],
    ids=["fp32", "bf16"],
)
def test_cross_framework_precision(jax_dtype, torch_dtype, atol, label):
    """JAX vs PyTorch 跨框架精度对比。"""
    rng = np.random.default_rng(42)
    input_data = rng.standard_normal((4, 8, HIDDEN_SIZE)).astype(np.float32)
    weight = rng.standard_normal(HIDDEN_SIZE).astype(np.float32)

    jax_out = run_jax(make_jax_model(HIDDEN_SIZE, NUM_GROUPS, EPSILON, weight), input_data, jax_dtype)
    torch_out = run_torch(make_torch_model(HIDDEN_SIZE, NUM_GROUPS, EPSILON, weight), input_data, torch_dtype)

    _print_diff(f"{label}: JAX vs PyTorch", jax_out, torch_out, atol)
    np.testing.assert_allclose(jax_out, torch_out, atol=atol, rtol=0)


def test_error_bound_verification():
    """用 fp64 ground truth 验证理论误差上界是否成立。"""
    single_bound = _n_ops * _eps_fp32  # 单框架理论上界

    max_rel_jax = 0.0
    max_rel_torch = 0.0

    for seed in range(100):
        rng = np.random.default_rng(seed)
        input_data = rng.standard_normal((4, 8, HIDDEN_SIZE)).astype(np.float32)
        weight = rng.standard_normal(HIDDEN_SIZE).astype(np.float32)

        fp64_out = numpy_group_rmsnorm_fp64(input_data, weight, NUM_GROUPS)
        jax_out = run_jax(make_jax_model(HIDDEN_SIZE, NUM_GROUPS, EPSILON, weight), input_data, jnp.float32).astype(
            np.float64
        )
        torch_out = run_torch(
            make_torch_model(HIDDEN_SIZE, NUM_GROUPS, EPSILON, weight), input_data, torch.float32
        ).astype(np.float64)

        # 相对误差：|impl - truth| / |truth|
        nonzero = np.abs(fp64_out) > 1e-12
        rel_jax = np.max(np.abs(jax_out[nonzero] - fp64_out[nonzero]) / np.abs(fp64_out[nonzero]))
        rel_torch = np.max(np.abs(torch_out[nonzero] - fp64_out[nonzero]) / np.abs(fp64_out[nonzero]))
        max_rel_jax = max(max_rel_jax, rel_jax)
        max_rel_torch = max(max_rel_torch, rel_torch)

    print(f"\n{'=' * 50}")
    print("  Error bound verification (100 trials)")
    print(f"{'=' * 50}")
    print(f"  theoretical bound: {single_bound:.6e}")
    print(f"  JAX  max rel err:  {max_rel_jax:.6e}")
    print(f"  Torch max rel err: {max_rel_torch:.6e}")
    print(f"  JAX  within bound: {max_rel_jax <= single_bound}")
    print(f"  Torch within bound: {max_rel_torch <= single_bound}")
    print(f"{'=' * 50}")

    np.testing.assert_array_less(max_rel_jax, single_bound, err_msg=f"JAX exceeded bound: {max_rel_jax:.6e}")
    np.testing.assert_array_less(max_rel_torch, single_bound, err_msg=f"Torch exceeded bound: {max_rel_torch:.6e}")
