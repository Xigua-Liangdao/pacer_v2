import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aide_clip" / "src"))

from clip_aide_emotion_train import ClipImageAdapter, StrongerClipImageAdapter, TemporalTransformerClipImageAdapter


def test_clip_adapter_identity_normalizes_input():
    x = torch.randn(4, 8)
    adapter = ClipImageAdapter(
        dim=8,
        device="cpu",
        hidden_dim=16,
        dropout=0.0,
        num_classes=3,
        num_prompts=5,
        adapter_mode="identity",
    )
    with torch.no_grad():
        adapter.input_proj.weight.zero_()
        adapter.input_proj.bias.zero_()
    out = adapter._adapt_image(x)
    assert torch.allclose(out, F.normalize(x, dim=-1), atol=1e-6)
    assert sum(p.numel() for p in adapter.adapter_parameters()) == 0
    assert sum(p.numel() for p in adapter.parameters()) == 1 + 3 * 5 + 3 + 3


def test_stronger_adapter_identity_normalizes_input():
    x = torch.randn(2, 8)
    adapter = StrongerClipImageAdapter(
        dim=8,
        device="cpu",
        hidden_dim=16,
        dropout=0.0,
        num_classes=2,
        num_prompts=4,
        adapter_mode="identity",
    )
    out = adapter._adapt_image(x)
    assert torch.allclose(out, F.normalize(x, dim=-1), atol=1e-6)
    assert sum(p.numel() for p in adapter.adapter_parameters()) == 0


def test_temporal_adapter_identity_keeps_temporal_pool_then_normalizes():
    x = torch.randn(3, 4, 8)
    adapter = TemporalTransformerClipImageAdapter(
        dim=8,
        device="cpu",
        hidden_dim=16,
        dropout=0.0,
        num_classes=2,
        num_prompts=4,
        num_frames=4,
        temporal_num_heads=4,
        temporal_num_layers=1,
        temporal_pool_mode="mean",
        temporal_module="mean_pool",
        adapter_mode="identity",
    )
    out = adapter._adapt_image(x)
    assert torch.allclose(out, F.normalize(x.mean(dim=1), dim=-1), atol=1e-6)
    assert sum(p.numel() for p in adapter.adapter_parameters()) == 0
