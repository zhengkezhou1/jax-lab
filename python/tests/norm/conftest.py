import sys
from pathlib import Path

# Add model/norm/ to sys.path so tests can import group_rms_norm_jax, group_rms_norm_torch.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model" / "norm"))
