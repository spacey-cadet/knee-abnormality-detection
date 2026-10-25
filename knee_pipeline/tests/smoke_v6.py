"""60-second smoke test for the v6 changes. Needs torch + timm (a Kaggle notebook has both), no GPU, no pretrained
weights, no data. Run BEFORE launching any long job:  python tests/smoke_v6.py
Checks: both heads give [B,12] finite logits and train; absent series cannot influence the output; the token head
really attends over all windows (changing one window changes the output); old v5 checkpoints still load into
head="series"; KneeDS applies geom codes exactly like geom.apply_study."""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import KneeNet
from geom import apply_study, tf_code

torch.manual_seed(0)
B, S, D, H = 2, 3, 12, 64
x = torch.rand(B, S, D, H, H) * 255
pres = torch.tensor([[1, 1, 1], [1, 0, 1]])
plane = torch.tensor([[0, 1, 2]] * B); fluid = torch.ones(B, S, dtype=torch.long); fat = torch.ones(B, S, dtype=torch.long)

for head in ("series", "tokens"):
    m = KneeNet("efficientnet_b0", pretrained=False, head=head, train_slices=8)
    m.train(); out = m(x, pres, plane, fluid, fat)
    assert out.shape == (B, 12) and torch.isfinite(out).all(), head
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.enc.parameters()), "no gradient reaches the encoder"
    # fresh BatchNorm running stats (mean 0 / var 1) make a random-init EfficientNet's activations vanish (~1e-6) in eval,
    # so ANY input change moves the logits by <1e-6. Calibrate the stats on real-scale input first so the checks are meaningful.
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None   # cumulative average
    with torch.no_grad():
        for _ in range(3): m(x, pres, plane, fluid, fat)
    m.eval()
    with torch.no_grad():
        base = m(x, pres, plane, fluid, fat)
        x2 = x.clone(); x2[1, 1] = torch.rand(D, H, H) * 255          # change the ABSENT series of study 1
        assert torch.allclose(base, m(x2, pres, plane, fluid, fat), atol=1e-5), f"{head}: absent series leaks"
        x3 = x.clone(); x3[0, 2, 5] = torch.rand(H, H) * 255          # change ONE window of a present series
        assert not torch.allclose(base[0], m(x3, pres, plane, fluid, fat)[0], atol=1e-6), f"{head}: input ignored"
    print(f"head={head}: forward/backward OK, absent-series masking OK, params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

# round-2 backbones: build WITHOUT weights, forward + backward at the real 160 px cache size
ARMS = ["convnext_tiny.in12k_ft_in1k", "efficientnet_b2", "vit_small_patch14_dinov2.lvd142m",
        "vit_small_patch16_224.dino", "deit_tiny_patch16_224.fb_in1k"]
xs = torch.rand(1, 3, 6, 160, 160) * 255
for bb in ARMS:
    mm = KneeNet(bb, pretrained=False, head="tokens", train_slices=4)
    mm.train(); o = mm(xs, torch.ones(1, 3, dtype=torch.long), torch.tensor([[0, 1, 2]]),
                       torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))
    assert o.shape == (1, 12) and torch.isfinite(o).all(), bb
    o.sum().backward()
    print(f"backbone {bb}: OK | vit_size {mm.vit_size} | params {sum(p.numel() for p in mm.parameters())/1e6:.1f}M")

# v5 checkpoints (no 'head' key) must still load into the default head
sd = KneeNet("efficientnet_b0", pretrained=False).state_dict()
KneeNet("efficientnet_b0", pretrained=False, head="series").load_state_dict(sd)
print("v5-style state_dict loads into head='series'")

# KneeDS geometry parity
import pandas as pd
from train import KneeDS
with tempfile.TemporaryDirectory() as d:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (3, 6, 8, 8)).astype(np.uint8)
    np.save(f"{d}/s1.npy", arr)
    from common import LABELS
    row = {"StudyInstanceUID": "s1", **{f"y_{l}": 0.0 for l in LABELS}, **{f"w_{l}": 1.0 for l in LABELS}}
    for j in range(3):
        row.update({f"present_{j}": 1, f"plane_{j}": j, f"fluid_{j}": 1, f"fat_{j}": 1})
    df = pd.DataFrame([row])
    codes = [tf_code([0, 1, 0, 0, 0, -1], 0), tf_code([-1, 0, 0, 0, 0, -1], 1), tf_code([1, 0, 0, 0, 1, 0], 2)]
    g = {"s1": (codes, "R")}
    got = KneeDS(df, d, 1, False, geom=g)[0][0].numpy().astype(np.uint8)
    want = apply_study(arr, [0, 1, 2], [1, 1, 1], codes, "R")
    assert np.array_equal(got, want) and not np.array_equal(got, arr)
    assert np.array_equal(KneeDS(df, d, 1, False)[0][0].numpy().astype(np.uint8), arr)
print("KneeDS geometry parity OK\nALL V6 SMOKE TESTS PASS")
