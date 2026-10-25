"""Labeler B via the Kaggle Benchmarks model proxy (uses the free Kaggle AI quota).

Run INSIDE a Kaggle Benchmarks notebook (https://www.kaggle.com/benchmarks/tasks/new).
Same prompt, cache, parsing and output format as labeler_llm.py, so B_*.csv files plug into
labeler_combine.py unchanged. Only the backend differs.

UNTESTED against the real proxy. The kaggle_benchmarks calls used are llm.prompt(text) -> str (from the
library's documented example); the model lookup is defensive. ALWAYS run `probe` first.

  python labeler_kbench.py probe --model google/gemini-3.7-flash
  python labeler_kbench.py label --train_csv train_gold.csv --out B_gold_gem37.csv --model google/gemini-3.7-flash
  python labeler_kbench.py label --train_csv train.csv --out B_full_gem37.csv --model ... --limit 500

Cache = <out>.cache.jsonl, keyed by study id only: use a NEW --out per model / prompt. Re-running the same
--out resumes (failed parses are retried).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# Works as a script AND when pasted/exec'd in a notebook cell (no __file__ there).
# In a notebook, set CODE_DIR below (or env KNEE_CODE) to the folder holding labeler_llm.py and common.py.
CODE_DIR = os.environ.get("KNEE_CODE", "")
if not CODE_DIR:
    try:
        CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
import labeler_llm as L  # noqa: E402


def get_llm(name):
    import kaggle_benchmarks as kb
    if name in (None, "", "default", os.environ.get("LLM_DEFAULT")):
        return kb.llm
    for attr in ("llms", "models"):
        reg = getattr(kb, attr, None)
        if reg is not None:
            try:
                return reg[name]
            except Exception:
                pass
    for fn in ("get_llm", "load_llm", "llm_by_name"):
        f = getattr(kb, fn, None)
        if callable(f):
            try:
                return f(name)
            except Exception:
                pass
    raise RuntimeError(f"Could not get model {name!r}. kaggle_benchmarks exposes: "
                       f"{[n for n in dir(kb) if not n.startswith('_')]}. Paste this list back.")


def as_text(r):
    if isinstance(r, str):
        return r
    for a in ("text", "content", "response"):
        if hasattr(r, a):
            return str(getattr(r, a))
    return str(r)


def flat_prompt(report):
    """llm.prompt takes one string, so system text + few-shot are flattened into it."""
    parts = [L.SYSTEM, "\nExamples:\n"]
    for txt, ans in L.FEWSHOT:
        parts.append(f"Report:\n{txt}\n\nJSON:\n{json.dumps(ans, ensure_ascii=False)}\n")
    parts.append(f"Now the real report. Reply with the JSON object only.\n\nReport:\n{str(report)[:6000]}\n\nJSON:")
    return "\n".join(parts)


class KbenchBackend:
    def __init__(self, model, workers=4):
        self.llm, self.workers, self.model = get_llm(model), workers, model
        self.calls = self.fails = 0

    def _one(self, report):
        self.calls += 1
        for attempt in range(4):
            try:
                return as_text(self.llm.prompt(flat_prompt(report)))
            except Exception as e:
                last = repr(e)[:200]
                time.sleep(2 ** attempt)
        self.fails += 1
        print("  call failed:", last)
        return ""

    def generate(self, reports):
        if self.workers <= 1:
            return [self._one(r) for r in reports]
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, reports))


def probe(model):
    llm = get_llm(model)
    print("model object:", type(llm), "| methods:", [m for m in dir(llm) if not m.startswith("_")][:30])
    t = time.time()
    a = as_text(llm.prompt("Remember the secret word PINEAPPLE. Reply only: ok"))
    b = as_text(llm.prompt("What secret word did I ask you to remember? If you do not know, reply exactly: UNKNOWN"))
    print(f"reply 1: {a!r}\nreply 2: {b!r}  ({time.time() - t:.1f}s)")
    if "PINEAPPLE" in b.upper():
        print("\n!! STATEFUL: the model sees earlier calls. Every call would resend the whole history and the "
              "quota would be burned fast. DO NOT run `label` yet; paste this output back.")
    else:
        print("\nOK: calls look independent (stateless).")
    r = as_text(llm.prompt(flat_prompt(L.FEWSHOT[1][0])))
    parsed, ok = L.parse_json(r)
    print("parse_ok on a few-shot report:", ok, "\nraw:", r[:300])
    print("expected:", json.dumps(L.FEWSHOT[1][1])[:200])
    print("\nNow read the quota page, note the value, and compare after a --limit 50 run.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--model", default=None)
    q = sub.add_parser("label")
    q.add_argument("--train_csv", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--prompt", default="v1", choices=list(L.PROMPTS))
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--limit", type=int, default=None, help="max NEW reports this run (quota cap)")
    q.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        return probe(a.model)
    L.set_prompt(a.prompt)
    df = pd.read_csv(a.train_csv)
    be = KbenchBackend(a.model, a.workers)
    res = L.run(df, be, a.out + ".cache.jsonl", chunk=a.chunk, limit=a.limit)
    res.to_csv(a.out, index=False)
    json.dump({"model": a.model, "backend": "kbench", "prompt": a.prompt,
               "prompt_sha1": hashlib.sha1(L.SYSTEM.encode()).hexdigest()[:12],
               "n": len(res), "labelled_ok": int(res.B_parse_ok.sum()),
               "calls_this_run": be.calls, "call_failures": be.fails},
              open(a.out + ".meta.json", "w"), indent=1)
    print("saved", a.out, res.shape, "| parsed ok:", int(res.B_parse_ok.sum()), "| call failures:", be.fails)


if __name__ == "__main__" and "__file__" in globals():
    main()
