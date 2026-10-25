import sys; sys.path.insert(0, "/home/claude/knee_pipeline")
from labeler_rules import label_report
# (text, {label: 'hi'|'lo'|'mid'|'any'})
CASES = [
 ("Complete tear of the anterior cruciate ligament. No joint effusion.", {"ACL":"hi","Effusion":"lo"}),
 ("The ACL is intact. Medial meniscus: horizontal tear of the posterior horn. Lateral meniscus is normal.", {"ACL":"lo","Medial Meniscus":"hi","Lateral Meniscus":"lo"}),
 ("No evidence of ACL tear or MCL sprain.", {"ACL":"lo","MCL":"lo"}),
 ("ACL tear, MCL intact.", {"ACL":"hi","MCL":"lo"}),
 ("Possible partial tear of the lateral meniscus.", {"Lateral Meniscus":"mid"}),
 ("A tear of the lateral meniscus cannot be excluded.", {"Lateral Meniscus":"mid"}),
 ("Small joint effusion. Baker's cyst present.", {"Effusion":"hi","Baker's":"hi"}),
 ("No effusion, synovitis or Baker cyst.", {"Effusion":"lo","Synovitis":"lo","Baker's":"lo"}),
 ("Bone marrow edema in the lateral femoral condyle consistent with contusion. No fracture.", {"Contusion":"hi","Fracture":"lo"}),
 ("Medial compartment chondromalacia with osteophytes. Lateral compartment cartilage is preserved.", {"Medial OA":"hi","Lateral OA":"lo"}),
 ("Patellofemoral osteoarthritis with cartilage loss of the trochlea.", {"PF OA":"hi"}),
 ("Tricompartmental osteoarthritis.", {"Medial OA":"hi","Lateral OA":"hi","PF OA":"hi"}),
 ("Degeneration of the medial meniscus without tear.", {"Medial Meniscus":"lo"}),
 ("Status post ACL reconstruction, graft intact.", {"ACL":"lo"}),
 # Spanish
 ("Rotura completa del ligamento cruzado anterior. Derrame articular moderado.", {"ACL":"hi","Effusion":"hi"}),
 ("Menisco interno con rotura del cuerno posterior. Menisco externo sin alteraciones.", {"Medial Meniscus":"hi","Lateral Meniscus":"lo"}),
 ("No se observa derrame articular ni quiste de Baker.", {"Effusion":"lo","Baker's":"lo"}),
 # French
 ("Rupture complète du ligament croisé antérieur. Pas d'épanchement articulaire.", {"ACL":"hi","Effusion":"lo"}),
 ("Fissure du ménisque interne. Ménisque externe normal.", {"Medial Meniscus":"hi","Lateral Meniscus":"lo"}),
 # German
 ("Kniegelenkserguss. Riss des Innenmeniskus. Kein Anhalt für eine Fraktur.", {"Effusion":"hi","Medial Meniscus":"hi","Fracture":"lo"}),
 ("Vorderes Kreuzband intakt. Keine Bakerzyste.", {"ACL":"lo","Baker's":"lo"}),
 ("Knochenmarködem im lateralen Femurkondylus.", {"Contusion":"hi"}),
 # Portuguese / Italian
 ("Rotura do menisco medial. Sem derrame articular.", {"Medial Meniscus":"hi","Effusion":"lo"}),
 ("Rottura del legamento crociato anteriore. Versamento articolare presente.", {"ACL":"hi","Effusion":"hi"}),
 # Russian / Chinese
 ("Разрыв переднего крестообразного связки. Выпот в полости сустава не выявлен.", {"ACL":"hi","Effusion":"lo"}),
 ("前交叉韧带完全撕裂。关节腔未见积液。", {"ACL":"hi","Effusion":"lo"}),
 ("内侧半月板撕裂,外侧半月板正常。", {"Medial Meniscus":"hi","Lateral Meniscus":"lo"}),
 # nothing
 ("", {"ACL":"lo","Effusion":"lo"}),
]
bad = 0
for text, exp in CASES:
    p, c, n = label_report(text)
    for lab, e in exp.items():
        v = p[lab]
        ok = (e == "hi" and v >= 0.7) or (e == "lo" and v <= 0.2) or (e == "mid" and 0.3 <= v <= 0.6)
        if not ok:
            bad += 1
            print(f"FAIL [{lab}] expected {e} got {v:.2f} | {text[:90]}")
print(f"{sum(len(e) for _,e in CASES)-bad}/{sum(len(e) for _,e in CASES)} checks passed")
