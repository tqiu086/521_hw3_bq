"""Faithfulness audits (structured, typicality) + intervention summary.  python audits.py"""
import transformers  # must precede decoder (which stubs `datasets`)
import itertools, json, math, os, random, re, sys
import numpy as np
from decoder import question_pool, parse_steps, norm, CONSTANTS
from run_eval import load_task, ROOT

# ----------------------------------------------------------- stats
def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    po = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= po * (1 + 1e-7)))

def report(name, items, fail_key, label):
    """items: list of dicts with 'correct' and fail_key (True = fails audit)."""
    n = len(items)
    f = [r for r in items if r[fail_key]]
    p_ = [r for r in items if not r[fail_key]]
    acc = lambda rs: 100 * sum(r["correct"] for r in rs) / len(rs) if rs else float("nan")
    pv = fisher_exact(sum(r["correct"] for r in p_), len(p_) - sum(r["correct"] for r in p_),
                      sum(r["correct"] for r in f), len(f) - sum(r["correct"] for r in f)) if f and p_ else float("nan")
    return f"| {label} | {name} | {n} | {100*len(f)/n:.1f} | {acc(p_):.1f} ({len(p_)}) | {acc(f):.1f} ({len(f)}) | {pv:.2g} |"

# ----------------------------------------------------------------------- (i) structured audits
OPS = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y if y else None]

def exprs(vals):
    """All values of binary expression trees over all permutations of vals using + - * /."""
    if len(vals) == 1:
        yield vals[0]; return
    seen = set()
    for perm in set(itertools.permutations(vals)):
        for split in range(1, len(perm)):
            for l in exprs(list(perm[:split])):
                for r in exprs(list(perm[split:])):
                    for op in OPS:
                        v = op(l, r)
                        if v is not None and v not in seen:
                            seen.add(v); yield v

def step_recomputes(inputs, out, consts):
    vals = [float(v) for v in inputs]
    out = float(out)
    ok = lambda v: abs(v - out) <= 0.01 * max(1.0, abs(out))
    if len(vals) == 1:   # one operand + one constant/pool number
        return ok(vals[0]) or any(ok(v) for c in consts for v in exprs([vals[0], c]))
    if len(vals) <= 3:   # any binary expression tree over any ordering
        return any(ok(v) for v in exprs(vals))
    # 4+ operands: operands in the written order, any operator string, Python precedence or left-to-right fold
    for ops in itertools.product("+-*/", repeat=len(vals) - 1):
        terms = [f"({v})" for v in vals]
        expr = "".join(t + o for t, o in zip(terms, ops)) + terms[-1]
        try:
            if ok(eval(expr)):
                return True
        except ZeroDivisionError:
            pass
        acc = vals[0]
        try:
            for o, v in zip(ops, vals[1:]):
                acc = {"+": acc + v, "-": acc - v, "*": acc * v, "/": acc / v}[o]
        except ZeroDivisionError:
            continue
        if ok(acc):
            return True
    return False

def structured(task, rows, items):
    res = []
    for r in rows:
        it = items[r["id"]]
        consts = [float(c) for c in question_pool(it["q"], constants=(task == "gsm8k"))]
        steps = r["steps"]
        has = len(steps) > 0
        a1 = not (has and r["provenance_ok"])                                  # input provenance
        a2 = not (has and all(step_recomputes(s["inputs"].values(), s["output"], consts) for s in steps))  # arithmetic recompute
        try:
            a3 = not (has and r["pred"] is not None and abs(float(steps[-1]["output"]) - float(r["pred"])) < 1e-6)  # answer = last step
        except ValueError:
            a3 = True
        res.append(dict(r, fail_prov=a1, fail_arith=a2, fail_final=a3))
    return res

# ------------------------------------------------------------------------------------(ii) typicality audit
def embed_names(names):
    import torch
    from run_eval import load_model
    model, tok = load_model()
    out = []
    with torch.no_grad():
        for nm in names:
            ids = tok(f"A step in a math solution is named: {nm.replace('_', ' ')}. This step computes", return_tensors="pt").to(model.device)
            h = model(**ids, output_hidden_states=True).hidden_states[20][0, -1].float().cpu().numpy()
            out.append(h)
    X = np.stack(out); X -= X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    del model
    return X

def kmeans(X, k, seed=0, iters=50):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0); C[j] /= np.linalg.norm(C[j]) + 1e-9
    return lab

def typicality(all_rows, K=6, q=0.2):
    names = sorted({s["name"] for rows in all_rows.values() for r in rows for s in r["steps"]})
    lab = dict(zip(names, kmeans(embed_names(names), K)))
    clusters = {}
    for nm, c in lab.items():
        clusters.setdefault(int(c), []).append(nm)
    out, sizes = {}, {c: len(v) for c, v in clusters.items()}
    for key, rows in all_rows.items():
        seqs = [["<s>"] + [f"c{lab[s['name']]}" for s in r["steps"]] + ["</s>"] if r["steps"] else None for r in rows]
        trans = [list(zip(s[:-1], s[1:])) if s else [] for s in seqs]
        cnt, ctx = {}, {}
        for t in trans:
            for a, b in t:
                cnt[(a, b)] = cnt.get((a, b), 0) + 1; ctx[a] = ctx.get(a, 0) + 1
        V = K + 1
        scores = []
        for t in trans:
            if not t:
                scores.append(None); continue
            own, ownc = {}, {}
            for a, b in t:                                     # leave-one-out counts
                own[(a, b)] = own.get((a, b), 0) + 1; ownc[a] = ownc.get(a, 0) + 1
            scores.append(np.mean([math.log((cnt[(a, b)] - own[(a, b)] + 1) / (ctx[a] - ownc[a] + V)) for a, b in t]))
        valid = [s for s in scores if s is not None]
        thr = np.quantile(valid, q)
        out[key] = [dict(r, fail_typ=(s < thr), typ_score=s) for r, s in zip(rows, scores) if s is not None]
    return out, clusters

# ------------------------------------------------------------------------------------- main
def main():
    lines = ["| task / method | audit | N | % fail | acc if pass (n) | acc if fail (n) | Fisher p |", "|---|---|---|---|---|---|---|"]
    all_rows, items = {}, {}
    for task in ["gsm8k", "bbh"]:
        items[task] = {it["id"]: it for it in load_task(task, 200)}
        for method in ["crane", "template"]:
            all_rows[(task, method)] = [json.loads(l) for l in open(f"results/{task}_{method}.jsonl", encoding="utf-8")]
    for (task, method), rows in all_rows.items():
        s = structured(task, rows, items[task])
        for key, nm in [("fail_prov", "S1 input provenance"), ("fail_arith", "S2 step arithmetic recomputed"), ("fail_final", "S3 answer = last step output")]:
            lines.append(report(nm, s, key, f"{task} / {method}"))
    typ, clusters = typicality(all_rows)
    for (task, method), rows in typ.items():
        lines.append(report("T1 typicality (bottom 20% log-prob)", rows, "fail_typ", f"{task} / {method}"))
    for task in ["gsm8k", "bbh"]:
        p = f"results/intervene_{task}.jsonl"
        if os.path.exists(p):
            rs = [json.loads(l) for l in open(p, encoding="utf-8")]
            rs = [dict(r, correct=r["orig_correct"], fail_int=r["unchanged"]) for r in rs]
            lines.append(report("I1 intervention (fail = answer unchanged)", rs, "fail_int", f"{task} / crane"))
    txt = "\n".join(lines)
    print(txt)
    open("results/audit_table.md", "w", encoding="utf-8").write(txt + "\n\nName clusters (k=6):\n" + "\n".join(
        f"c{c}: {len(v)} names, e.g. {', '.join(v[:8])}" for c, v in sorted(clusters.items())) + "\n")
    for c, v in sorted(clusters.items()):
        print(f"c{c}: {len(v)} names, e.g. {', '.join(v[:8])}")

if __name__ == "__main__":
    main()
