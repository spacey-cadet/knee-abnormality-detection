"""Time ONE backbone with the real micro-batch (bs x 3 series x 16 windows of 3 x 160 x 160) and print one JSON line.
  python bench_backbones.py --arm efficientnet_b0 [--bs 4]
Reports: train seconds per micro-batch (fwd+bwd+step, AMP), peak GPU memory, inference seconds per study
(24 windows x 3 series, no grad), parameter count. One arm per process so an OOM cannot poison the next arm."""
import argparse, json, statistics, sys, time
import torch
from model import KneeNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = dict(arm=a.arm, bs=a.bs)
    try:
        m = KneeNet(a.arm, pretrained=False, head="tokens", train_slices=16).to(dev).train()
        res["params_m"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
        S, D, H = 3, 24, 160
        x = torch.rand(a.bs, S, D, H, H, device=dev) * 255
        pres = torch.ones(a.bs, S, dtype=torch.long, device=dev)
        plane = torch.tensor([[0, 1, 2]] * a.bs, device=dev)
        one = torch.ones(a.bs, S, dtype=torch.long, device=dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler()
        ts = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.steps):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda"):
                out = m(x, pres, plane, one, one)
            scaler.scale(out.float().mean()).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); ts.append(time.time() - t)
        res["train_s"] = round(statistics.median(ts[2:]), 4)
        res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del opt, out; torch.cuda.empty_cache()
        m.eval(); x1 = x[:1]; p1, pl1, o1 = pres[:1], plane[:1], one[:1]
        ti = []
        with torch.no_grad(), torch.autocast("cuda"):
            for i in range(6):
                torch.cuda.synchronize(); t = time.time(); m(x1, p1, pl1, o1, o1); torch.cuda.synchronize(); ti.append(time.time() - t)
        res["infer_s_per_study"] = round(statistics.median(ti[2:]), 4)
        res["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        res["status"] = "oom"
    except Exception as e:
        res["status"] = "error: " + repr(e)[:200]
    print("BENCH_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
