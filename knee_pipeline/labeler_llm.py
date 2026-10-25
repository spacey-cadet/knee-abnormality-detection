"""Labeler B - LLM-based report labeler (multilingual, few-shot, JSON output, resumable cache).

Backends
  hf         : local instruction-tuned model via transformers. RECOMMENDED - competition data never leaves
               your notebook. On Kaggle attach a model from Kaggle Models and pass --model_path.
  anthropic  : Claude via API. Needs internet + ANTHROPIC_API_KEY. It sends competition reports to a third
               party: CHECK THE COMPETITION RULES before using it.
  mock       : deterministic stand-in (uses Labeler A) for testing the pipeline without a model.

Output per report: for each label one of present | absent | uncertain | not_mentioned.
Mapped to probability/confidence below. Results are cached to JSONL so an interrupted run can resume.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import LABELS

VALUES = ["present", "absent", "uncertain", "not_mentioned"]
PROB = {"present": 0.95, "uncertain": 0.50, "absent": 0.03, "not_mentioned": 0.05}
CONF = {"present": 0.85, "uncertain": 0.50, "absent": 0.85, "not_mentioned": 0.40}

DEFINITIONS = """\
ACL: anterior cruciate ligament injury (tear, rupture, sprain). An intact ACL or an intact reconstruction graft = absent.
MCL: medial collateral ligament injury (sprain or tear of any grade).
Medial Meniscus: tear of the medial (internal) meniscus. Degeneration/intrasubstance signal without a tear = absent.
Lateral Meniscus: tear of the lateral (external) meniscus. Same rule as above.
Medial OA: osteoarthritis / cartilage loss / chondromalacia / osteophytes in the MEDIAL tibiofemoral compartment.
Lateral OA: same, LATERAL tibiofemoral compartment.
PF OA: same, PATELLOFEMORAL compartment (patella, trochlea).
Effusion: joint effusion / excess joint fluid.
Synovitis: synovial inflammation / thickening / hypertrophy.
Baker's: Baker's (popliteal / gastrocnemius-semimembranosus) cyst. Other cysts do not count.
Contusion: bone contusion / bone bruise / traumatic bone-marrow edema.
Fracture: any fracture (including avulsion, stress, osteochondral).
"""

SYSTEM_V1 = (
    "You extract structured findings from knee MRI radiology reports written in any language. "
    "Only use what the report text says; never infer findings that are not stated. "
    "For each finding answer with exactly one of: present, absent, uncertain, not_mentioned.\n"
    "- present: the report affirms the finding.\n"
    "- absent: the report explicitly says the finding is absent or the structure is normal/intact.\n"
    "- uncertain: hedged (possible, suspected, cannot be excluded, questionable).\n"
    "- not_mentioned: the report does not address it.\n"
    "Reply with a single JSON object only, keys exactly as listed, no other text.\n\n"
    "Finding definitions:\n" + DEFINITIONS
)

# v2: changes motivated by the gold error analysis (collateral-ligament mix-ups, 'bone marrow edema' misses,
# hedged/mild findings, contradictory sentences). UNTESTED on the model: compare v1 vs v2 on a gold dev split
# (`--prompt v2`, separate --out/--cache) before adopting it.
EXTRA_RULES_V2 = (
    "Additional rules:\n"
    "- Judge only the findings/impression text; ignore the clinical history/indication.\n"
    "- MCL means the MEDIAL (tibial) collateral ligament only. A lateral/fibular collateral (LCL/FCL) injury is NOT an MCL finding; "
    "say MCL is absent only if the report says the MEDIAL collateral is intact.\n"
    "- Contusion counts bone contusion, bone bruise and bone-marrow/bony/subchondral edema. Muscle or soft-tissue edema does NOT count.\n"
    "- Mild, minimal, trace, small, low-grade or grade 1 findings are still present. Use uncertain only for explicit doubt "
    "(possible, suspected, cannot be excluded) or when the report contradicts itself about that finding.\n"
    "- Effusion includes hemarthrosis. Synovitis includes synovial thickening/hypertrophy/reaction.\n"
    "- If one sentence says 'no fracture' (or any 'no X') and another names X, answer uncertain.\n"
    "- Silence about a structure is not_mentioned, never absent.\n"
)
SYSTEM_V2 = SYSTEM_V1 + "\n" + EXTRA_RULES_V2
SYSTEM = SYSTEM_V1            # default stays v1 so existing runs reproduce; selected by --prompt
PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}

FEWSHOT = [
    ("Complete tear of the ACL. Horizontal tear of the posterior horn of the medial meniscus. "
     "Lateral meniscus normal. Small joint effusion. No Baker's cyst. Mild chondromalacia of the medial compartment.",
     {"ACL": "present", "MCL": "not_mentioned", "Medial Meniscus": "present", "Lateral Meniscus": "absent",
      "Medial OA": "present", "Lateral OA": "not_mentioned", "PF OA": "not_mentioned", "Effusion": "present",
      "Synovitis": "not_mentioned", "Baker's": "absent", "Contusion": "not_mentioned", "Fracture": "not_mentioned"}),
    ("Ligamento cruzado anterior íntegro. Posible rotura del menisco externo. No se observa derrame articular. "
     "Edema óseo en el cóndilo femoral lateral. Sin fracturas.",
     {"ACL": "absent", "MCL": "not_mentioned", "Medial Meniscus": "not_mentioned", "Lateral Meniscus": "uncertain",
      "Medial OA": "not_mentioned", "Lateral OA": "not_mentioned", "PF OA": "not_mentioned", "Effusion": "absent",
      "Synovitis": "not_mentioned", "Baker's": "not_mentioned", "Contusion": "present", "Fracture": "absent"}),
    ("Vorderes Kreuzband intakt. Innenmeniskus mit degenerativen Veränderungen ohne Riss. Retropatellare Arthrose. "
     "Ausgeprägter Kniegelenkserguss mit Synovialitis. Bakerzyste.",
     {"ACL": "absent", "MCL": "not_mentioned", "Medial Meniscus": "absent", "Lateral Meniscus": "not_mentioned",
      "Medial OA": "not_mentioned", "Lateral OA": "not_mentioned", "PF OA": "present", "Effusion": "present",
      "Synovitis": "present", "Baker's": "present", "Contusion": "not_mentioned", "Fracture": "not_mentioned"}),
]


def build_messages(report):
    msgs = []
    for txt, ans in FEWSHOT:
        msgs.append({"role": "user", "content": f"Report:\n{txt}\n\nJSON:"})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"Report:\n{str(report)[:6000]}\n\nJSON:"})
    return msgs


def parse_json(text):
    """Return {label: value}; missing/invalid -> not_mentioned. Second return: parse ok flag."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    out, ok = {l: "not_mentioned" for l in LABELS}, False
    if not m:
        return out, ok
    try:
        d = json.loads(m.group(0))
    except Exception:
        return out, ok
    ok = True
    for l in LABELS:
        v = str(d.get(l, "not_mentioned")).strip().lower().replace(" ", "_")
        out[l] = v if v in VALUES else "not_mentioned"
    return out, ok



def set_prompt(version):
    global SYSTEM
    SYSTEM = PROMPTS[version]


def chat_prompt(tok, report):
    """Chat-template prompt. Qwen3: thinking is switched off (it would emit <think> text and waste tokens).
    Templates without a system role (e.g. Gemma 2) get the system text merged into the first user turn."""
    msgs = build_messages(report)
    try:
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM}] + msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except Exception:
        first = dict(msgs[0]); first["content"] = SYSTEM + "\n\n" + first["content"]
        return tok.apply_chat_template([first] + msgs[1:], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)


# --------------------------------------------------------------------------- backends
class MockBackend:
    """Uses Labeler A to fake an LLM answer (for testing the plumbing only)."""

    def __init__(self):
        from labeler_rules import label_report
        self.f = label_report

    def generate(self, reports):
        outs = []
        for r in reports:
            p, _, _ = self.f(r)
            d = {l: ("present" if p[l] > .8 else "uncertain" if .3 < p[l] < .7 else
                     "absent" if p[l] < .04 else "not_mentioned") for l in LABELS}
            outs.append(json.dumps(d))
        return outs


class HFBackend:
    def __init__(self, model_path, load_4bit=False, max_new_tokens=260, batch_size=8, temperature=0.0, seed=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(torch_dtype=torch.float16, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw).eval()
        self.max_new, self.bs = max_new_tokens, batch_size
        self.temperature, self.seed = float(temperature), int(seed)

    def generate(self, reports):
        outs = []
        for i in range(0, len(reports), self.bs):
            chunk = reports[i:i + self.bs]
            prompts = [chat_prompt(self.tok, r) for r in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            gk = dict(do_sample=False)
            if self.temperature > 0:                      # sampled run (for vote-fraction confidence)
                self.torch.manual_seed(self.seed + i)     # reproducible per batch
                gk = dict(do_sample=True, temperature=self.temperature, top_p=0.95)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new,
                                          pad_token_id=self.tok.pad_token_id, **gk)
            for g in gen:
                outs.append(self.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs


class AnthropicBackend:
    def __init__(self, model="claude-haiku-4-5-20251001", workers=8):
        import anthropic
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        self.model, self.workers = model, workers

    def _one(self, report):
        for attempt in range(5):
            try:
                r = self.client.messages.create(model=self.model, max_tokens=400, system=SYSTEM,
                                                messages=build_messages(report))
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception as e:  # rate limits / transient errors
                time.sleep(2 ** attempt)
        return ""

    def generate(self, reports):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, reports))


def make_backend(name, **kw):
    if name == "mock":
        return MockBackend()
    if name == "hf":
        return HFBackend(kw["model_path"], kw.get("load_4bit", False), batch_size=kw.get("batch_size", 8),
                         temperature=kw.get("temperature", 0.0), seed=kw.get("seed", 0))
    if name == "anthropic":
        return AnthropicBackend(kw.get("model") or "claude-haiku-4-5-20251001")
    raise ValueError(name)


# --------------------------------------------------------------------------- driver
def run(df, backend, cache_path, chunk=32, limit=None, text_col="Report", retry_failed=True):
    cache_path = Path(cache_path)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if retry_failed and not r.get("parse_ok"):
                    continue                      # re-label rows whose JSON failed to parse
                done[r["StudyInstanceUID"]] = r
            except Exception:
                pass
    todo = df[~df.StudyInstanceUID.isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"cached: {len(done)} | to label: {len(todo)}")
    with open(cache_path, "a") as f:
        for i in range(0, len(todo), chunk):
            part = todo.iloc[i:i + chunk]
            raw = backend.generate(part[text_col].fillna("").astype(str).tolist())
            for sid, txt in zip(part.StudyInstanceUID, raw):
                parsed, ok = parse_json(txt)
                rec = {"StudyInstanceUID": sid, "parse_ok": ok, **parsed}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[sid] = rec
            f.flush()
            print(f"  labeled {min(i + chunk, len(todo))}/{len(todo)}")
    rows = []
    for sid in df.StudyInstanceUID:
        r = done.get(sid)
        row = {"StudyInstanceUID": sid, "B_parse_ok": bool(r and r.get("parse_ok"))}
        for l in LABELS:
            v = r[l] if r else "not_mentioned"
            row[f"B_{l}"] = PROB[v]
            row[f"Bc_{l}"] = CONF[v] if (r and r.get("parse_ok")) else 0.0
            row[f"Bv_{l}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf", "anthropic", "mock"])
    ap.add_argument("--model_path", default=None, help="hf: local model dir / hub id")
    ap.add_argument("--model", default=None, help="anthropic: model name")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--prompt", default="v1", choices=list(PROMPTS), help="v1 = original; v2 = revised rules (test first)")
    ap.add_argument("--temperature", type=float, default=0.0, help=">0 => sampled run (use with --seed, K runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_retry_failed", action="store_true")
    a = ap.parse_args()
    set_prompt(a.prompt)
    df = pd.read_csv(a.train_csv)
    be = make_backend(a.backend, model_path=a.model_path, model=a.model, load_4bit=a.load_4bit, batch_size=a.batch_size,
                      temperature=a.temperature, seed=a.seed)
    res = run(df, be, a.cache or (a.out + ".cache.jsonl"), limit=a.limit, retry_failed=not a.no_retry_failed)
    res.to_csv(a.out, index=False)
    import hashlib
    json.dump({"model": a.model_path or a.model, "backend": a.backend, "load_4bit": a.load_4bit, "prompt": a.prompt,
               "prompt_sha1": hashlib.sha1(SYSTEM.encode()).hexdigest()[:12], "temperature": a.temperature,
               "seed": a.seed, "n": len(res), "parse_failures": int((~res.B_parse_ok).sum())},
              open(a.out + ".meta.json", "w"), indent=1)
    print("saved", a.out, res.shape, "| parse failures:", int((~res.B_parse_ok).sum()))
