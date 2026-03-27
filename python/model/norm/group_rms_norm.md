# GroupRMSNorm：官方 PyTorch vs 我们的 JAX 实现对比

## 构造函数

| 行为 | 官方 PyTorch | 我们的 JAX |
|---|---|---|
| 可学习权重 | `nn.Parameter(torch.ones(hidden_size))` | `nnx.Param(jnp.ones(hidden_size))` |
| 分组数 | `group_norm_size` | `num_groups` |
| 精度保护 | `eps=1e-6` | `epsilon=1e-6` |
| 整除校验 | `assert hidden_size % group_norm_size == 0` | `assert hidden_size % num_groups == 0` |

- `nn.Parameter`（PyTorch）和 `nnx.Param`（Flax）作用相同：声明一个**可学习参数**，在训练时梯度会更新它
- 权重初始化为全 1，意味着归一化刚开始时不做任何缩放
- 官方参数名 `group_norm_size` 实际表示的是**分组数量**（不是每组的大小），我们用 `num_groups` 更清晰

## Forward 逐行对比

假设输入 shape 是 `(batch, seq_len, 8192)`，`num_groups=8`。

### 第 1 步：保存原始信息

| 官方 PyTorch | 我们的 JAX |
|---|---|
| `input_dtype = hidden_states.dtype` | `orig_dtype = hidden_states.dtype` |
| `input_shape = hidden_states.size()` | `orig_shape = hidden_states.shape` |

完全一样：保存原始的数据类型和形状，以便最后恢复。

### 第 2 步：reshape 分组

**官方 PyTorch：**
```python
group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
hidden_states = hidden_states.view(group_input_shape)
```

**我们的 JAX：**
```python
hidden_states = hidden_states.reshape(*orig_shape[:-1], self.num_groups, -1)
```

- 两者做的事情完全相同：把最后一维 `(8192,)` 拆成 `(8, 1024)`，即 8 个组，每组 1024 个元素
- PyTorch 的写法是手动算出 `group_input_shape` 再 `.view()`；JAX 用 `-1` 让框架自动推断组大小
- `.view()` 和 `.reshape()` 在连续内存时行为一致，都是**零拷贝**操作，只改变数据的"视角"，不移动内存中的数据

### 第 3 步：提升到 float32

| 官方 PyTorch | 我们的 JAX |
|---|---|
| `hidden_states = hidden_states.to(torch.float32)` | `hidden_states = hidden_states.astype(jnp.float32)` |

完全等价。如果模型使用 bfloat16 推理，归一化的中间计算需要用 float32 以避免精度丢失。

### 第 4 步：计算方差（RMS 的核心）

**官方 PyTorch：**
```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

**我们的 JAX：**
```python
variance = jnp.mean(lax.square(hidden_states), axis=-1, keepdims=True)
```

- 这一步计算的是**均方值（mean of squares）**，不是传统意义上的"方差"（方差需要先减均值）。这就是 **RMS**Norm 和 Layer**Norm** 的关键区别：RMSNorm 不减均值，只用平方的均值
- `.pow(2)` 等价于 `lax.square()`：对每个元素求平方
- `.mean(-1, keepdim=True)`：在最后一维（每组 1024 个元素）上求平均，`keepdim=True` 保持维度以便广播
- 计算结果 shape：`(batch, seq_len, 8, 1)` — 每个组得到一个标量

### 第 5 步：归一化

**官方 PyTorch：**
```python
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
```

**我们的 JAX：**
```python
hidden_states = hidden_states * lax.rsqrt(variance + self.epsilon)
```

完全等价。`rsqrt(x) = 1 / sqrt(x)`，即：x_norm = x × (1 / sqrt(mean(x²) + ε))

- `epsilon`（1e-6）是为了防止除零：如果某组的所有值恰好为 0，没有 epsilon 就会出现 `1/0 = inf`
- 用 `rsqrt` 而不是 `1/sqrt` 是因为 **rsqrt 在 GPU/TPU 上有专用硬件指令**，速度更快
- 广播机制：`variance` 的 shape 是 `(..., 8, 1)`，会自动广播到 `(..., 8, 1024)` 与 `hidden_states` 逐元素相乘

### 第 6 步：恢复形状 + 应用可学习权重

**官方 PyTorch：**
```python
return self.weight * hidden_states.to(input_dtype).view(input_shape)
```

**我们的 JAX：**
```python
hidden_states = hidden_states.astype(orig_dtype).reshape(orig_shape)
return self.weight[...] * hidden_states
```

- 先把类型转回原始精度（如 bfloat16），再把 shape 从 `(..., 8, 1024)` 恢复为 `(..., 8192)`
- `self.weight` 的 shape 是 `(8192,)`，恢复成 `(..., 8192)` 后才能与之相乘（广播）
- JAX 中 `self.weight[...]` 的 `[...]` 是 Flax 的惯用写法，从 `nnx.Param` 容器中取出底层的 `jax.Array`
- 官方 PyTorch 中 `nn.Parameter` 可以直接参与运算，不需要额外解包

## 结论

**两个实现在数学上完全一致。** 每一步操作都能一一对应：

```
保存dtype/shape → reshape分组 → 提升float32 → 平方求均值 → rsqrt归一化 → 恢复dtype/shape → 乘权重
```

唯一的"差异"是命名风格（`group_norm_size` vs `num_groups`）和框架的 API 写法不同，但计算逻辑完全相同。

## 实验验证

我们编写了跨框架测试（`group_rms_norm_test.py`），用相同的输入分别通过 PyTorch 和 JAX 模型，验证两者的一致性。

### 测试项

| 测试 | 说明 |
|---|---|
| `test_shape` | JAX 输出 shape 与输入 shape 一致 |
| `test_group_independence` | 修改一个 group 的输入不影响其他 group 的输出 |
| `test_learnable_scale` | 可学习权重正确参与计算（与 numpy 参考实现对比） |
| `test_output_shape_consistent` | JAX 和 PyTorch 输出 shape 相同，且等于输入 shape |
| `test_output_values_match_default_weight` | 默认权重（全 1）下两者输出数值一致 |
| `test_output_values_match_random_weight` | 随机权重下两者输出数值一致 |
| `test_output_values_match_2d_input` | 2D 输入 `(batch, hidden)` 下 shape 和数值均一致 |

### 精度差异

两者在数学上等价，但由于底层数学库和浮点运算顺序不同（如 `rsqrt` 的硬件实现、浮点累加顺序），**不是 bit-exact 相等**：

| 指标 | 数值 |
|---|---|
| 最大绝对误差 | ~9.5×10⁻⁷ |
| 平均绝对误差 | ~1.8×10⁻⁸ |
| `np.array_equal` 严格相等 |
| `assert_allclose(rtol=1e-6)` |

测试中使用 `rtol=1e-6` 作为容差，验证的是数学等价性而非 bit-level 一致性。这个精度差异在 float32 下完全正常。
