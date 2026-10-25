"""Inference: decode test studies on the fly, run every fold model, rank-average, write submission.csv.
Fallback for unreadable studies: training prior per label (from labels_soft.csv) instead of 0.5."""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset
from cache_data import process_study
from common import LABELS, find_data_dir
from model import KneeNet


class TestDS(Dataset):
    def __init__(self, root, ts, studies, D, size, K):
        self.root, self.ts, self.studies, self.D, self.size, self.K = root, ts, studies, D, size, K
    def __len__(self): return len(self.studies)
    def __getitem__(self, i):
        s = self.studies[i]
        try:
            r = process_study(self.root, s, self.ts[self.ts.StudyInstanceUID == s], self.D, self.size, self.size, self.K)
            ok = int(r["present"].sum() > 0)
        except Exception:
            S = 3 * self.K
            r = dict(arr=np.zeros((S, self.D, self.size, self.size), np.uint8), present=np.zeros(S, np.int8),
                     plane=np.repeat(np.arange(3), self.K), fluid=np.zeros(S, np.int8), fat=np.zeros(S, np.int8)); ok = 0
        t = lambda k: torch.as_tensor(r[k]).long()
        return torch.from_numpy(r["arr"]).float(), t("present"), t("plane"), t("fluid"), t("fat"), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--labels", default=None, help="labels_soft.csv for fallback priors")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--D", type=int, default=24)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--average", default="rank", choices=["rank", "prob"])
    a = ap.parse_args()
    data = Path(a.data_dir) if a.data_dir else find_data_dir()
    sub = pd.read_csv(data / "sample_submission.csv")
    ts = pd.read_csv(data / "test_series.csv")
    studies = pd.read_csv(data / "test.csv").StudyInstanceUID.tolist()
    ts["n_slices"] = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(Path(a.models_dir).glob("fold*.pt"))
    assert ckpts, f"no fold*.pt in {a.models_dir}"
    models = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        m = KneeNet(ck["backbone"], False, None); m.load_state_dict(ck["state"]); models.append(m.to(dev).eval()); K = ck["K"]
    dl = DataLoader(TestDS(data / "test_series", ts, studies, a.D, a.size, K), batch_size=a.bs, num_workers=a.workers)
    P, OK, t0 = [], [], time.time()
    with torch.no_grad():
        for x, pres, plane, fluid, fat, ok in dl:
            x, pres, plane, fluid, fat = [v.to(dev) for v in (x, pres, plane, fluid, fat)]
            preds = [torch.sigmoid(m(x, pres, plane, fluid, fat).float()).cpu().numpy() for m in models]
            P.append(np.stack(preds)); OK.append(ok.numpy())
            print(f"{sum(len(o) for o in OK)}/{len(studies)} | {time.time() - t0:.0f}s", flush=True)
    P, OK = np.concatenate(P, 1), np.concatenate(OK).astype(bool)           # [M, N, 12]
    if a.average == "rank":
        R = np.stack([[rankdata(P[m, :, j]) / len(studies) for j in range(12)] for m in range(len(models))])  # [M,12,N]
        final = R.mean(0).T
    else:
        final = P.mean(0)
    prior = np.full(12, 0.5)
    if a.labels:
        L = pd.read_csv(a.labels); prior = np.array([L[f"y_{l}"].mean() for l in LABELS])
    final[~OK] = prior
    out = pd.DataFrame(final, columns=LABELS); out.insert(0, "StudyInstanceUID", studies)
    out = sub[["StudyInstanceUID"]].merge(out, on="StudyInstanceUID", how="left")[list(sub.columns)]
    assert list(out.columns) == list(sub.columns) and out[LABELS].notna().all().all() and ((out[LABELS] >= 0) & (out[LABELS] <= 1)).all().all()
    out.to_csv(a.out, index=False)
    print("saved", a.out, out.shape, "| unreadable studies:", int((~OK).sum()), f"| {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
