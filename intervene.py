"""Intervention test: edit the output of one step of a crane output, continue generation from there, compare answers.
python intervene.py  -> results/intervene_<task>.jsonl
"""
import json, os, random, re, sys
from run_eval import *          # imports model loading, prompts, scoring
from decoder import RECORD_RE, norm

def edit_value(out, rng):
    v = float(out)
    for _ in range(20):
        new = v + rng.choice([-1, 1]) * rng.randint(2, 9) * (10 ** rng.randint(0, 1))
        if new != v:
            return norm(new)

def main():
    model, tok = load_model()
    dec = CraneDecoder(model, tok, os.path.join(ROOT, "step_grammar.lark"))
    for task in ["gsm8k", "bbh"]:
        items = {it["id"]: it for it in load_task(task, 200)}
        rows = [json.loads(l) for l in open(f"results/{task}_crane.jsonl", encoding="utf-8")]
        path = f"results/intervene_{task}.jsonl"
        done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                if r["id"] in done or r["n_steps"] == 0:
                    continue
                rng = random.Random(r["id"])
                recs = list(RECORD_RE.finditer(r["output"]))
                k = rng.randrange(len(recs))                  # step to edit
                m = recs[k]
                old = m.group(3)
                new = edit_value(old, rng)
                head = r["output"][:m.start(3)] + new + ">>"  # everything up to the edited record's end
                prefix_outputs = [x.group(3) for x in recs[:k]] + [new]
                it = items[r["id"]]
                q_pool = question_pool(it["q"], constants=(task == "gsm8k"))
                prompt = tok.apply_chat_template(build_messages("crane", task, it["q"]), add_generation_prompt=True)
                cont, _ = dec.generate(prompt, q_pool, 700, prefix_ids=tok(head, add_special_tokens=False).input_ids,
                                       prefix_outputs=prefix_outputs)
                pred_new = extract_answer(head + cont)
                row = {"id": r["id"], "step_idx": k, "n_steps": r["n_steps"], "is_last_step": k == r["n_steps"] - 1,
                       "old": old, "new": new, "pred_orig": r["pred"], "pred_new": pred_new,
                       "unchanged": pred_new is not None and r["pred"] is not None and abs(float(pred_new) - float(r["pred"])) < 1e-6,
                       "orig_correct": r["correct"], "continuation": cont}
                f.write(json.dumps(row) + "\n"); f.flush()
                print(task, r["id"], k, old, "->", new, "| ans", r["pred"], "->", pred_new, flush=True)

if __name__ == "__main__":
    main()
