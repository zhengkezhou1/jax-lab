# AGENTS 指引

## 背景
这是我在 [JAX](https://github.com/jax-ml/jax) 上进行的对于 LLM 模型 (Transformer) 接入的实验仓库:
- 用 JAX 复现 PyTorch 组件并验证精度一致性

## 环境配置
使用 `conda activate jax-lab` 激活环境

## 编码规范
- **每次编写或修改 Python 文件后，必须立即使用 ruff 进行格式化和检查：** `make lint`。在进入下一步之前，确保对每个新建或修改的 Python 文件都执行此操作。
- 生产代码中禁止使用 `assert`。
- JAX 和 PyTorch 的双实现文件分别以 `*_jax.py` 和 `*_torch.py` 命名。

## 测试规范
- 为新行为添加测试——覆盖成功、失败和边界情况。
- 使用 pytest 模式，不使用 `unittest.TestCase`。
- 对多组相似输入使用 `@pytest.mark.parametrize`。