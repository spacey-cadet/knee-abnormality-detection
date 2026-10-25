"""Labeler A - multilingual rule-based labeler for knee MRI reports.

For every report and each of the 12 labels it returns a probability and a confidence:
  - present (affirmed)            -> high probability
  - hedged ("possible", ...)      -> 0.5
  - negated ("no tear", ...)      -> low probability
  - not mentioned                 -> small prior, low confidence

IMPORTANT
- Lexicons cover en, es, fr, de, pt, it, ru, zh. They were written from general knowledge, NOT reviewed
  by native-speaker radiologists. Run the audit (labeler_combine.py audit) per language and extend LEXICON
  for the languages you actually see in the data (the EDA notebook lists them).
- Negation handling is heuristic (character windows, clause splitting). Expect errors on long or unusual
  sentences; that is why labels stay soft and are combined with Labeler B.
"""
import argparse
import re
import unicodedata

import pandas as pd

from common import LABELS

# --------------------------------------------------------------------------- text normalisation


def fold(text):
    """NFKC + lowercase + strip accents on Latin letters only (keeps Cyrillic/CJK intact)."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    out = []
    for ch in text:
        if ord(ch) < 0x250:
            out.append("".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out).replace("ß", "ss")


_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


def rx(*alts):
    """Compile alternatives. A leading '~' means: no word-boundary before (German compounds).
    CJK alternatives never get a boundary."""
    parts = []
    for a in alts:
        if a.startswith(r"~") or _CJK.match(a):
            parts.append("(?:" + a.lstrip(r"~") + ")")
        else:
            parts.append(r"(?<!\w)(?:" + a + ")")
    return re.compile("|".join(parts))


# --------------------------------------------------------------------------- lexicon (all languages merged)

STRUCT = {
    "acl": rx(r"acl(?!\w)", r"lca(?!\w)", r"vkb(?!\w)", r"пкс(?!\w)", r"anterior cruciate", r"cruzado anterior",
              r"croise anterieur", r"crociato anteriore", r"vordere\w* kreuzband", r"передн\w* крестообразн",
              "前交叉韧带"),
    "mcl": rx(r"mcl(?!\w)", r"medial collateral", r"tibial collateral", r"collateral medial", r"collateral tibial",
              r"colateral medial", r"collaterale mediale", r"ligamento lateral interno", r"ligament lateral interne",
              r"lli(?!\w)", r"lcm(?!\w)", r"~innenband", r"mediale\w* kollateral\w*",
              r"медиальн\w* коллатеральн\w*", "内侧副韧带"),
    "menisc": rx(r"menisc\w*", r"menisq\w*", r"~menisk\w*", r"менис\w*", "半月板"),
}
MENISC_PLURAL = rx(r"menisci", r"meniscos", r"menisques", r"menisken", r"менисков", r"менисци",
                   r"menisc\w* (?:and|y|e|et|und) menisc\w*")
SIDE_MED = rx(r"medial\w*", r"interno", r"interne", r"internal", r"innen\w*", r"~innenmenisk", r"медиальн\w*", "内侧")
SIDE_LAT = rx(r"lateral\w*", r"externo", r"externe", r"external", r"aussen\w*", r"~aussenmenisk", r"латеральн\w*", "外侧")
SIDE_PF = rx(r"~patell\w*", r"~patelo\w*", r"rotul\w*", r"troclea\w*", r"trochle\w*", r"femoropatel\w*",
             r"пателл\w*", r"надколенник\w*", "髌", "滑车")
MULTI_COMP = rx(r"tricompartiment\w*", r"tricompartment\w*", r"multicompartiment\w*", r"multicompartment\w*",
                r"three compartments", r"todos los compartimentos", r"pangonartros\w*", "三间室")

TEAR = rx(r"tear\w*", r"torn(?!\w)", r"rupture\w*", r"ruptura\w*", r"rotura\w*", r"desgarr\w*", r"disrupt\w*",
          r"discontinu\w*", r"avuls\w*", r"sprain\w*", r"lacerat\w*", r"laceraz\w*", r"rottura", r"rotture",
          r"fissur\w*", r"dechirure\w*", r"~riss\w*", r"~ruptur\w*", r"lesion\w*", r"lesao", r"lesoes",
          r"повреждени\w*", r"разрыв\w*", r"надрыв\w*", "撕裂", "断裂", "损伤", "拉伤", "扭伤")
GRAFT = rx(r"graft", r"reconstruct\w*", r"plasti\w*", r"ligamentoplast\w*", r"transplant\w*",
           r"postoperat\w*", r"post-operat\w*", r"postquirurg\w*", "重建", "移植")

OA_STRONG = rx(r"osteoarthr\w*", r"osteoartr\w*", r"~arthros\w*", r"artros\w*", r"gonartros\w*", r"gonarthros\w*",
               r"chondromalac\w*", r"condromalac\w*", r"chondromalaz\w*", r"chondropath\w*", r"condropat\w*",
               r"cartilag\w*\s+(?:loss|damage|thinning|defect\w*|wear|erosion\w*|degenerat\w*|lesion\w*|fissur\w*|denud\w*|adelgaz\w*)",
               r"(?:loss|thinning|damage|defect\w*|wear) of (?:the )?(?:articular )?cartilag\w*",
               r"perdida de cartilago", r"usure (?:du )?cartilag\w*", r"~knorpelschad\w*", r"~knorpelverlust",
               r"~knorpeldefekt\w*", r"~knorpellasion\w*", r"pincement", r"osteophyt\w*", r"osteofit\w*",
               r"joint space narrowing", r"pinzamiento", r"gelenkspaltverschmaler\w*", r"остеоартроз\w*",
               r"гонартроз\w*", r"артроз\w*", r"хондромаляц\w*", r"хондропати\w*", r"остеофит\w*",
               "骨关节炎", "骨性关节炎", "软骨损伤", "软骨变性", "软骨磨损", "软骨缺损", "骨赘", "关节间隙变窄")
OA_WEAK = rx(r"degenerative (?:changes?|joint)", r"cambios degenerativos", r"alteraciones degenerativas",
             r"remaniements? degeneratifs?", r"degenerative veranderung\w*", r"alterazioni degenerative",
             r"alteracoes degenerativas", r"дегенеративн\w*", "退行性", "退变")

EFFUSION = rx(r"~effusion\w*", r"derrame\w*", r"epanchement\w*", r"~erguss\w*", r"versamento", r"выпот\w*", "积液", "渗出")
MILD = rx(r"trace", r"minimal\w*", r"physiolog\w*", r"small", r"slight\w*", r"mild\w*", r"leve", r"ligero",
          r"minimo", r"fisiolog\w*", r"gering\w*", r"leicht\w*", r"petit\w*", r"pequen\w*", r"piccolo", r"lieve",
          r"незначительн\w*", r"небольш\w*", "少量", "微量")
SYNOVITIS = rx(r"~synovit\w*", r"sinovit\w*", r"synovialit\w*",
               r"synovial (?:thickening|hypertroph\w*|proliferat\w*|inflammat\w*)",
               r"sinovial\w* (?:engros\w*|hipertrof\w*)", r"epaississement synovial", r"синовит\w*", "滑膜炎", "滑膜增厚")
BAKER = rx(r"~baker", r"popliteal cyst", r"poplite\w* cyst", r"quiste poplit\w*", r"kyste poplite\w*",
           r"cisto poplit\w*", r"cisti poplit\w*", r"~kniekehlenzyst\w*", r"киста бейкер\w*", "腘窝囊肿", "贝克囊肿")
CONTUSION = rx(r"~contusion\w*", r"contusione", r"contusao", r"bone bruis\w*", r"bruis\w*", r"marrow (?:edema|oedema)",
               r"bone marrow (?:edema|oedema|lesion\w*)", r"edema (?:oseo|de medula osea)", r"edema osseo",
               r"oedeme osseux", r"~knochenmarkodem\w*", r"~kontusion\w*", r"ушиб\w*", r"контузи\w*",
               r"отек\w* костн\w*", "骨挫伤", "骨髓水肿", "骨挫裂")
FRACTURE = rx(r"~fractur\w*", r"~fratur\w*", r"~frattur\w*", r"~fraktur\w*", r"segond", r"перелом\w*", "骨折")

ANY_PATH = re.compile("|".join(r.pattern for r in [TEAR, OA_STRONG, EFFUSION, SYNOVITIS, BAKER, CONTUSION, FRACTURE]))

# cues -----------------------------------------------------------------------------------------
NEG_PRE = rx(r"no(?!\w)", r"not(?!\w)", r"without", r"absent", r"absence of", r"negative for", r"free of", r"neither",
             r"nor(?!\w)", r"rule[sd]? out", r"lack of", r"sin(?!\w)", r"ausencia de", r"ausente", r"ni(?!\w)",
             r"ningun\w*", r"pas d[e']", r"absence de", r"sans(?!\w)", r"aucun\w*", r"kein\w*", r"ohne(?!\w)",
             r"nicht(?!\w)", r"weder", r"frei von", r"sem(?!\w)", r"nao(?!\w)", r"nenhum\w*", r"senza(?!\w)",
             r"assenza", r"non(?!\w)", r"nessun\w*", r"нет(?!\w)", r"без(?!\w)", r"не(?!\w)", r"отсутств\w*",
             "无", "未见", "未显示", "未发现", "没有", "排除")
NORMAL = rx(r"normal\w*", r"intact\w*", r"unremarkable", r"preserved", r"healthy", r"intacto", r"intacta",
            r"integr\w*", r"conservad\w*", r"regelrecht", r"unauffallig", r"indemne", r"respectad\w*", r"целостн\w*",
            r"сохранен\w*", r"нормальн\w*", "正常", "完整")
POST_NEG = rx(r"absent", r"ausente", r"negative", r"not (?:seen|identified|visuali[sz]ed|demonstrated|evident|present|appreciated|observed)",
              r"no se (?:observa|aprecia|visualiza|identifica|evidencia)\w*", r"non (?:visualizzat\w*|evidente)",
              r"nao (?:se )?(?:observa|visualiza|identifica)\w*", r"pas (?:de|visible)",
              r"nicht (?:nachweisbar|abgrenzbar|erkennbar|sichtbar|darstellbar)", r"не (?:выявл\w*|определ\w*|визуализ\w*)",
              "未见", "未显示")
HEDGE_NEGLIKE = rx(r"cannot (?:be )?(?:exclude\w*|ruled out|rule out)", r"can not be (?:excluded|ruled out)",
                   r"not (?:be )?exclude\w*", r"no se descarta", r"no puede descartarse", r"no se puede descartar",
                   r"nao se pode excluir", r"nao pode ser excluid\w*", r"ne peut (?:etre )?(?:exclu|ecarte)\w*",
                   r"nicht (?:auszuschliessen|ausgeschlossen)", r"non si esclude", r"не исключ\w*", "不除外")
HEDGE = rx(r"possible", r"possibly", r"probabl\w*", r"suspect\w*", r"suggest\w*", r"question\w*", r"equivocal",
           r"indeterminate", r"may (?:represent|reflect|be|indicate)", r"could (?:represent|reflect|be)", r"likely",
           r"versus", r"posible", r"sospech\w*", r"sugier\w*", r"possivel", r"possiveis", r"suspeit\w*",
           r"suspicion", r"doute", r"evoque\w*", r"moglich\w*", r"wahrscheinlich", r"verdacht", r"fraglich\w*",
           r"vereinbar", r"passend", r"possibile", r"probabile", r"sospett\w*", r"вероятн\w*", r"возможн\w*",
           r"подозрен\w*", "可能", "考虑", "怀疑", "待排")
ASSERT = rx(r"there (?:is|are)", r"shows?", r"demonstrat\w*", r"reveal\w*", r"noted", r"consistent with",
            r"es gibt", r"zeigt", r"besteht", r"есть(?!\w)", r"определяется", r"выявляется", "存在", "可见", "显示")
CLAUSE_SPLIT = re.compile(
    r"(?<!\d)[.](?!\d)|[;!?\n\r。；！？]|(?<!\w)(?:but|however|although|except|whereas|aber|jedoch|allerdings|pero|"
    r"sin embargo|excepto|mais|cependant|sauf|mas|porem|exceto|ma|tuttavia|tranne|но|однако|кроме)(?!\w)|但是|但|然而")
POST_CUT_PAIR = re.compile(r",|(?<!\w)(?:and|or|y|o|e|et|ou|und|oder|и|или)(?!\w)|[、，和及]")
POST_CUT_STRUCT = re.compile(r",|[，]")

PRE_W, POST_W = 60, 55

# probabilities / confidences (soft labels)
P_POS, C_POS = 0.95, 0.90
P_MILD, C_MILD = 0.75, 0.60
P_UNC, C_UNC = 0.50, 0.60
P_NEG, C_NEG = 0.03, 0.85
P_NONE, C_NONE = 0.05, 0.40
P_CONFLICT, C_CONFLICT = 0.70, 0.50
P_SIDE_UNKNOWN, C_SIDE_UNKNOWN = 0.50, 0.30
P_GRAFT, C_GRAFT = 0.50, 0.40

# --------------------------------------------------------------------------- core logic


def _flags(clause, s, e, pair):
    """Return ('neg'|'unc'|'pos', mild) for a target span [s, e) inside a clause."""
    pre = clause[max(0, s - PRE_W):s]
    post = clause[e:e + POST_W]
    # negation scope in front only starts after the last assertive marker ("there is ... no ...")
    last_assert = None
    for m in ASSERT.finditer(pre):
        last_assert = m.end()
    if last_assert is not None:
        pre = pre[last_assert:]
    cut = (POST_CUT_PAIR if pair else POST_CUT_STRUCT).search(post)
    post_near = post[:cut.start()] if cut else post
    hedge_neg = bool(HEDGE_NEGLIKE.search(pre) or HEDGE_NEGLIKE.search(post_near))
    pre2, post2 = HEDGE_NEGLIKE.sub(" ", pre), HEDGE_NEGLIKE.sub(" ", post_near)
    negated = bool(NEG_PRE.search(pre2) or NORMAL.search(pre2[-25:]) or POST_NEG.search(post2)
                   or NORMAL.search(post2[:30]))
    if not pair and not negated and cut is None:
        pass
    if negated:
        return "neg"
    if hedge_neg or HEDGE.search(pre2) or HEDGE.search(post2):
        return "unc"
    return "pos"


def _nearest(matches, pos):
    best, bd = None, 10 ** 9
    for m in matches:
        d = min(abs(m.start() - pos), abs(m.end() - pos))
        if d < bd:
            best, bd = m, d
    return best, bd


def _mentions_in_clause(c):
    """Yield (label, status, prob_override) for a clause."""
    out = []
    # ---- ACL / MCL
    for key, lab in (("acl", "ACL"), ("mcl", "MCL")):
        structs = list(STRUCT[key].finditer(c))
        if not structs:
            continue
        tears = list(TEAR.finditer(c))
        used = False
        for t in tears:
            st, d = _nearest(structs, t.start())
            if st is None or d > 45:
                continue
            # the tear belongs to the nearest structure among ACL/MCL/menisc (avoid stealing)
            others = [m for k2, r2 in STRUCT.items() if k2 != key for m in r2.finditer(c)]
            ot, od = _nearest(others, t.start())
            if ot is not None and od < d:
                continue
            used = True
            status = _flags(c, t.start(), t.end(), pair=True)
            if lab == "ACL" and status != "neg" and GRAFT.search(c):
                out.append((lab, "graft", None))
            else:
                out.append((lab, status, None))
        if not used:
            for st in structs:
                if _flags(c, st.start(), st.end(), pair=False) == "neg" and (
                        NORMAL.search(c[max(0, st.start() - 25):st.start()]) or
                        NORMAL.search(c[st.end():st.end() + 40].split(",")[0]) or
                        POST_NEG.search(c[st.end():st.end() + 40])):
                    # only trust structure-only negation when an explicit "normal/intact" style cue exists
                    post_txt = c[st.end():st.end() + 60].split(",")[0]
                    if not ANY_PATH.search(post_txt):
                        out.append((lab, "neg", None))
    # ---- menisci
    menis = list(STRUCT["menisc"].finditer(c))
    if menis:
        tears = list(TEAR.finditer(c))
        others = [m for k2, r2 in STRUCT.items() if k2 != "menisc" for m in r2.finditer(c)]
        meds, lats = list(SIDE_MED.finditer(c)), list(SIDE_LAT.finditer(c))
        plural = bool(MENISC_PLURAL.search(c))

        def sides_for(m):
            lo, hi = m.start() - 25, m.end() + 25
            has_m = [x for x in meds if lo <= x.start() <= hi]
            has_l = [x for x in lats if lo <= x.start() <= hi]
            if has_m and has_l:
                # nearest wins unless both are directly attached ("medial and lateral menisci")
                if plural:
                    return ["Medial Meniscus", "Lateral Meniscus"], False
                dm, dl = min(abs(x.start() - m.start()) for x in has_m), min(abs(x.start() - m.start()) for x in has_l)
                return (["Medial Meniscus"] if dm <= dl else ["Lateral Meniscus"]), False
            if has_m:
                return ["Medial Meniscus"], False
            if has_l:
                return ["Lateral Meniscus"], False
            if plural:
                return ["Medial Meniscus", "Lateral Meniscus"], False
            return ["Medial Meniscus", "Lateral Meniscus"], True

        used_any = False
        for t in tears:
            mm, d = _nearest(menis, t.start())
            if mm is None or d > 45:
                continue
            ot, od = _nearest(others, t.start())
            if ot is not None and od < d:
                continue
            used_any = True
            status = _flags(c, t.start(), t.end(), pair=True)
            labs, unknown = sides_for(mm)
            for lab in labs:
                out.append((lab, "sideunk" if (unknown and status == "pos") else status, None))
        if not used_any:
            for mm in menis:
                post_txt = c[mm.end():mm.end() + 60].split(",")[0]
                if ANY_PATH.search(post_txt):
                    continue
                if (NORMAL.search(c[max(0, mm.start() - 25):mm.start()]) or NORMAL.search(post_txt[:35])
                        or POST_NEG.search(post_txt)):
                    labs, _ = sides_for(mm)
                    for lab in labs:
                        out.append((lab, "neg", None))
    # ---- OA / cartilage
    strong = list(OA_STRONG.finditer(c))
    weak = list(OA_WEAK.finditer(c)) if not any(STRUCT[k].search(c) for k in ("acl", "mcl", "menisc")) else []
    for m in strong + weak:
        lo, hi = m.start() - 45, m.end() + 45
        labs = set()
        meniscus_spans = [x for x in STRUCT["menisc"].finditer(c)]

        def attached_to_menisc(x):
            return any(0 <= ms.start() - x.end() <= 14 or 0 <= x.start() - ms.end() <= 2 for ms in meniscus_spans)

        for x in SIDE_MED.finditer(c):
            if lo <= x.start() <= hi and not attached_to_menisc(x):
                labs.add("Medial OA")
        for x in SIDE_LAT.finditer(c):
            if lo <= x.start() <= hi and not attached_to_menisc(x):
                labs.add("Lateral OA")
        for x in SIDE_PF.finditer(c):
            if lo <= x.start() <= hi:
                labs.add("PF OA")
        if MULTI_COMP.search(c[max(0, lo):hi]) or MULTI_COMP.search(c):
            labs |= {"Medial OA", "Lateral OA", "PF OA"}
        if labs:
            status = _flags(c, m.start(), m.end(), pair=True)
            for lab in labs:
                out.append((lab, status, None))
    # ---- single-term findings
    for lab, r in (("Effusion", EFFUSION), ("Synovitis", SYNOVITIS), ("Baker's", BAKER),
                   ("Contusion", CONTUSION), ("Fracture", FRACTURE)):
        for m in r.finditer(c):
            status = _flags(c, m.start(), m.end(), pair=True)
            mild = lab == "Effusion" and status == "pos" and bool(MILD.search(c[max(0, m.start() - 25):m.start() + 5]))
            out.append((lab, "mild" if mild else status, None))
    return out


def label_report(text):
    """-> (probs, confs, counts) dictionaries keyed by label."""
    t = fold(text if isinstance(text, str) else "")
    clauses = [c for c in CLAUSE_SPLIT.split(t) if c and c.strip()]
    ev = {l: [] for l in LABELS}
    for c in clauses:
        for lab, status, _ in _mentions_in_clause(c):
            ev[lab].append(status)
    probs, confs, counts = {}, {}, {}
    for lab in LABELS:
        s = ev[lab]
        n_pos = s.count("pos")
        n_mild = s.count("mild")
        n_neg = s.count("neg")
        n_unc = s.count("unc") + s.count("graft") + s.count("sideunk")
        counts[lab] = dict(pos=n_pos + n_mild, neg=n_neg, unc=n_unc)
        if n_pos and n_neg:
            p, c = P_CONFLICT, C_CONFLICT
        elif n_pos:
            p, c = P_POS, C_POS
        elif n_mild and n_neg:
            p, c = P_CONFLICT, C_CONFLICT
        elif n_mild:
            p, c = P_MILD, C_MILD
        elif s.count("unc"):
            p, c = P_UNC, C_UNC
        elif s.count("sideunk"):
            p, c = P_SIDE_UNKNOWN, C_SIDE_UNKNOWN
        elif s.count("graft"):
            p, c = P_GRAFT, C_GRAFT
        elif n_neg:
            p, c = P_NEG, C_NEG
        else:
            p, c = P_NONE, C_NONE
        probs[lab], confs[lab] = p, c
    return probs, confs, counts


def run(df, text_col="Report"):
    rows = []
    for sid, rep in zip(df["StudyInstanceUID"], df[text_col].fillna("")):
        p, c, n = label_report(rep)
        r = {"StudyInstanceUID": sid}
        for l in LABELS:
            r[f"A_{l}"] = p[l]
            r[f"Ac_{l}"] = c[l]
            r[f"An_{l}"] = n[l]["pos"] * 100 + n[l]["neg"] * 10 + n[l]["unc"]  # compact mention counts
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.train_csv)
    res = run(d)
    res.to_csv(a.out, index=False)
    print("saved", a.out, res.shape)
