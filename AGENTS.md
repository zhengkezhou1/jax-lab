# AGENTS 指引

## 背景
这是我在 [JAX](https://github.com/jax-ml/jax) 上进行的对于 LLM 模型 (Transformer) 接入的实验仓库:
- 用 JAX 复现 PyTorch 组件并验证精度一致性

## 环境配置
使用 `conda activate jax-lab` 激活环境

## 编码规范
- **每次编写或修改 Python 文件后，必须立即使用 ruff 进行格式化和检查：** `make lint`。在进入下一步之前，确保对每个新建或修改的 Python 文件都执行此操作。
- 生产代码中禁止使用 `assert`。
- JAX 生产代码放在 `python/model/` 下，PyTorch 参考实现放在对应测试目录的 `conftest.py` 中。
- Commit message 遵循 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 规范。

## 测试规范
- 为新行为添加测试——覆盖成功、失败和边界情况。
- 使用 pytest 模式，不使用 `unittest.TestCase`。
- 对多组相似输入使用 `@pytest.mark.parametrize`。

### 精度对比规范
1. **理论误差先行** — 在编写测试之前，先分析计算图中各步骤在目标数据类型（bf16、f32）下的理论误差上界。
2. **绝对误差与相对误差并行** — 测试必须同时验证 atol 和 rtol。atol 兜底小值区域，rtol 衡量大值区域的精度。使用 `np.testing.assert_allclose(atol=..., rtol=...)` 同时设置。
3. **宽松起步** — 初始测试使用基于理论分析的宽松容差，确保实现的基本正确性。
4. **经验收紧** — 通过大量随机采样（≥100 次）找到实测最大误差，乘以安全系数（3-5x）作为新的容差阈值。
5. **输出统计信息** — 测试输出应包含 max/mean × abs/rel 四个维度，便于判断误差分布是否集中。