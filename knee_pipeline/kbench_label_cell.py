# Standalone Kaggle Benchmarks labeler cell. No pipeline dataset needed: prompt, few-shot, parser and
# resumable cache are inlined from labeler_llm.py (v5), so B_*.csv match the pipeline's format.
# Paste into ONE cell of a Benchmarks notebook, run it, then call probe(...) / label(...) in the next cells.
# UNTESTED against the real proxy (llm.prompt(text) -> str comes from the library's documented example).
import json, os, re, time, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

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



# ---------------------------------------------------------------- Kaggle Benchmarks backend
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
    parts = [SYSTEM, "\nExamples:\n"]
    for txt, ans in FEWSHOT:
        parts.append(f"Report:\n{txt}\n\nJSON:\n{json.dumps(ans, ensure_ascii=False)}\n")
    parts.append(f"Now the real report. Reply with the JSON object only.\n\nReport:\n{str(report)[:6000]}\n\nJSON:")
    return "\n".join(parts)


class KbenchBackend:
    def __init__(self, model, workers=4):
        self.llm, self.workers, self.model = get_llm(model), workers, model
        self.calls = self.fails = 0

    def _one(self, report):
        self.calls += 1
        last = ""
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


def probe(model=None):
    llm = get_llm(model)
    print("model object:", type(llm), "| methods:", [m for m in dir(llm) if not m.startswith("_")][:30])
    t = time.time()
    a = as_text(llm.prompt("Remember the secret word PINEAPPLE. Reply only: ok"))
    b = as_text(llm.prompt("What secret word did I ask you to remember? If you do not know, reply exactly: UNKNOWN"))
    print(f"reply 1: {a!r}\nreply 2: {b!r}  ({time.time() - t:.1f}s)")
    if "PINEAPPLE" in b.upper():
        print("\n!! STATEFUL: the model sees earlier calls. Every call would resend the whole history and the "
              "quota would be burned fast. DO NOT run label() yet; paste this output back.")
    else:
        print("\nOK: calls look independent (stateless).")
    r = as_text(llm.prompt(flat_prompt(FEWSHOT[1][0])))
    parsed, ok = parse_json(r)
    print("parse_ok on a few-shot report:", ok, "\nraw:", r[:300])
    print("expected:", json.dumps(FEWSHOT[1][1])[:200])
    print("\nNow read the quota page, note the value, and compare after a small label(..., limit=50) run.")


def label(train_csv, out, model, prompt="v1", workers=4, limit=None, chunk=32):
    """Resumable. Re-running the same `out` continues (failed parses retried). New `out` per model/prompt.
    limit = max NEW reports this call (quota cap)."""
    set_prompt(prompt)
    df = pd.read_csv(train_csv)
    be = KbenchBackend(model, workers)
    res = run(df, be, out + ".cache.jsonl", chunk=chunk, limit=limit)
    res.to_csv(out, index=False)
    json.dump({"model": model, "backend": "kbench", "prompt": prompt,
               "prompt_sha1": hashlib.sha1(SYSTEM.encode()).hexdigest()[:12],
               "n": len(res), "labelled_ok": int(res.B_parse_ok.sum()),
               "calls_this_run": be.calls, "call_failures": be.fails},
              open(out + ".meta.json", "w"), indent=1)
    print("saved", out, res.shape, "| parsed ok:", int(res.B_parse_ok.sum()), "| call failures:", be.fails)
    return res
