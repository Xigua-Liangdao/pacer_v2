#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"

if "$PYTHON_BIN" - <<PYTEST_CHECK
try:
    import pytest  # noqa: F401
except Exception:
    raise SystemExit(1)
PYTEST_CHECK
then
  "$PYTHON_BIN" -m pytest -q
else
  echo "pytest is not installed; running lightweight adapter smoke check instead."
  "$PYTHON_BIN" - <<PYSMOKE
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

root = Path.cwd()
sys.path.insert(0, str(root / "aide_clip" / "src"))
from clip_aide_emotion_train import ClipImageAdapter, StrongerClipImageAdapter, TemporalTransformerClipImageAdapter

x = torch.randn(4, 8)
adapter = ClipImageAdapter(dim=8, device="cpu", hidden_dim=16, dropout=0.0, num_classes=3, num_prompts=5, adapter_mode="identity")
with torch.no_grad():
    adapter.input_proj.weight.zero_()
    adapter.input_proj.bias.zero_()
assert torch.allclose(adapter._adapt_image(x), F.normalize(x, dim=-1), atol=1e-6)
assert sum(p.numel() for p in adapter.adapter_parameters()) == 0

strong = StrongerClipImageAdapter(dim=8, device="cpu", hidden_dim=16, dropout=0.0, num_classes=2, num_prompts=4, adapter_mode="identity")
assert torch.allclose(strong._adapt_image(x[:2]), F.normalize(x[:2], dim=-1), atol=1e-6)

t = torch.randn(3, 4, 8)
temporal = TemporalTransformerClipImageAdapter(dim=8, device="cpu", hidden_dim=16, dropout=0.0, num_classes=2, num_prompts=4, num_frames=4, temporal_num_heads=4, temporal_num_layers=1, temporal_pool_mode="mean", temporal_module="mean_pool", adapter_mode="identity")
assert torch.allclose(temporal._adapt_image(t), F.normalize(t.mean(dim=1), dim=-1), atol=1e-6)
print("adapter smoke check passed")
PYSMOKE
fi
