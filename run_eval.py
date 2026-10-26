"""
Run CRANE-style decoder, unconstrained CoT and fixed-template (SSRM-style) prompt on GSM8K and BBH.

python run_eval.py --methods crane cot template --tasks gsm8k bbh --n 200
Outputs: results/<task>_<method>.jsonl  (one line per problem, resumable)
"""
import argparse, json, os, random, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from decoder import CraneDecoder, question_pool, parse_steps, norm

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "qwen2.5-7b-instruct")
TEMPLATE_NAMES = {"get_data", "compute", "final_answer"}

# ----------------------------------------------------------------------------- data
def load_task(task, n, seed=0):
    if task == "gsm8k":
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "gsm8k_test.jsonl"), encoding="utf-8")]
        items = [{"q": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for r in rows]
    else:
        d = json.load(open(os.path.join(ROOT, "data", "multistep_arithmetic_two.json"), encoding="utf-8"))
        items = [{"q": r["input"], "gold": r["target"]} for r in d["examples"]]
    idx = random.Random(seed).sample(range(len(items)), n)
    return [dict(items[i], id=i) for i in idx]

# ----------------------------------------------------------------------------- prompts
DEMO = {
    "gsm8k": ("A shop sells pens in packs of 12. Ana buys 5 packs and gives away 8 pens. How many pens does she have left?",
              [("Ana buys 5 packs with 12 pens each.", "total_pens", "pens_per_pack=12, packs=5", "60"),
               ("She gives away 8 pens.", "pens_left", "total=60, given_away=8", "52")], "52"),
    "bbh": ("((2 + 3 * 4) - (1 + 5)) =",
            [("First 3 * 4.", "multiply", "a=3, b=4", "12"), ("Then 2 + 12.", "add", "a=2, b=12", "14"),
             ("Then 1 + 5.", "add", "a=1, b=5", "6"), ("Finally 14 - 6.", "subtract", "a=14, b=6", "8")], "8"),
}
TEMPLATE_DEMO = {
    "gsm8k": [("get_data", "pens_per_pack=12", "12"), ("get_data", "packs=5", "5"), ("compute", "a=12, b=5", "60"),
              ("get_data", "given_away=8", "8"), ("compute", "a=60, b=8", "52"), ("final_answer", "value=52", "52")],
    "bbh": [("get_data", "a=3, b=4", "12"), ("compute", "a=2, b=12", "14"), ("compute", "a=1, b=5", "6"),
            ("compute", "a=14, b=6", "8"), ("final_answer", "value=8", "8")],
}
SYS = {
    "cot": "Solve the problem step by step. End with 'The answer is N' where N is a number.",
    "crane": ("Solve the problem by reasoning step by step in plain text. Whenever you read a value or do a calculation, "
              "record it as a step record: <<step name(arg1=value1, arg2=value2) -> result>>. You choose the step name "
              "and the argument names (snake_case) and as many steps as you need. Every argument value must be a number "
              "from the problem or the result of an earlier step. End with 'The answer is N' where N is a number."),
    "template": ("Solve the problem using ONLY these fixed steps, written as <<step name(arg=value, ...) -> result>>: "
                 "get_data (read a value from the problem), compute (one arithmetic operation on numbers from the problem "
                 "or earlier results), final_answer (the result). Do not use other step names. "
                 "End with 'The answer is N' where N is a number."),
}

def build_messages(method, task, q):
    method = "crane" if method == "crane_free" else method 
    ex_q, ex_steps, ex_ans = DEMO[task]
    if method == "cot":
        resp = " ".join(s[0] for s in ex_steps) + f" The answer is {ex_ans}."
    elif method == "crane":
        resp = "\n".join(f"{s[0]} <<step {s[1]}({s[2]}) -> {s[3]}>>" for s in ex_steps) + f"\nThe answer is {ex_ans}."
    else:
        resp = "\n".join(f"<<step {n}({a}) -> {o}>>" for n, a, o in TEMPLATE_DEMO[task]) + f"\nThe answer is {ex_ans}."
    return [{"role": "system", "content": SYS[method]}, {"role": "user", "content": ex_q},
            {"role": "assistant", "content": resp}, {"role": "user", "content": q}]

# ------------------------------------------------------------------- scoring
def extract_answer(text):
    ms = re.findall(r"answer is\s*:?\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    if not ms:
        ms = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not ms:
        return None
    return ms[-1].replace(",", "").rstrip(".")

def correct(pred, gold):
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False

def score(method, task, item, text):
    pool = question_pool(item["q"], constants=(task == "gsm8k"))
    rows, prov = parse_steps(text, pool)
    pred = extract_answer(text)
    parsed = len(rows) > 0 and pred is not None
    if method == "template":
        parsed = parsed and all(r["name"] in TEMPLATE_NAMES for r in rows)
    return {"id": item["id"], "gold": item["gold"], "pred": pred, "correct": correct(pred, item["gold"]),
            "n_steps": len(rows), "parsed": parsed, "provenance_ok": bool(rows) and prov, "steps": rows, "output": text}

# ----------------------------------------------------------------------------------- main
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map={"": 0}, torch_dtype=torch.bfloat16)
    return model.eval(), tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["crane", "cot", "template"])
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "bbh"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=700)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    model, tok = load_model()
    dec = CraneDecoder(model, tok, os.path.join(ROOT, "step_grammar.lark"))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for task in a.tasks:
        items = load_task(task, a.n)
        for method in a.methods:
            path = os.path.join(ROOT, "results", f"{a.tag}{task}_{method}.jsonl")
            done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
            todo = [it for it in items if it["id"] not in done]
            print(f"[{task}/{method}] {len(done)} done, {len(todo)} to go", flush=True)
            t0 = time.time()
            with open(path, "a", encoding="utf-8") as f:
                if method == "crane":
                    for k, it in enumerate(todo):
                        text_ids = tok.apply_chat_template(build_messages(method, task, it["q"]), add_generation_prompt=True)
                        text, st = dec.generate(text_ids, question_pool(it["q"], constants=(task == "gsm8k")), a.max_new)
                        r = score(method, task, it, text); r.update(st)
                        f.write(json.dumps(r) + "\n"); f.flush()
                        print(f"  {k+1}/{len(todo)} correct={r['correct']} steps={r['n_steps']} {time.time()-t0:.0f}s", flush=True)
                else:
                    for b in range(0, len(todo), a.bs):
                        batch = todo[b:b + a.bs]
                        prompts = [tok.apply_chat_template(build_messages(method, task, it["q"]), add_generation_prompt=True, tokenize=False) for it in batch]
                        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
                        for it, o in zip(batch, out):
                            text = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True)
                            f.write(json.dumps(score(method, task, it, text)) + "\n")
                        f.flush()
                        print(f"  {b+len(batch)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
