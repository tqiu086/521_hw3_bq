"""Print result tables from results/*.jsonl."""
import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
rows = []
for task in ["gsm8k", "bbh"]:
    for method in ["crane", "crane_free", "cot", "template"]:
        p = os.path.join(ROOT, "results", f"{task}_{method}.jsonl")
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p, encoding="utf-8")]
        n = len(rs)
        if n == 0:
            continue
        pct = lambda k: 100 * sum(bool(r[k]) for r in rs) / n
        rows.append((task, method, n, pct("parsed"), pct("provenance_ok"), pct("correct"),
                     sum(r["n_steps"] for r in rs) / n))
hdr = f"{'task':7}{'method':10}{'N':>5}{'parsed %':>10}{'prov-ok %':>11}{'acc %':>8}{'steps/q':>9}"
print(hdr); print("-" * len(hdr))
for t, m, n, a, b, c, s in rows:
    print(f"{t:7}{m:10}{n:5}{a:10.1f}{b:11.1f}{c:8.1f}{s:9.2f}")
    