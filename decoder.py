"""CRANE-style decoder: free text, switching to SynCode-constrained step records.

Free mode      : plain greedy decoding, no mask.
Constrained    : entered when the text ends with the start delimiter "<<step"
"""
import re
import sys
import types
import torch

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))
from syncode.grammar_mask.logits_processor import SyncodeLogitsProcessor
from syncode.parsers.grammars import Grammar

START = "<<step"
RECORD_RE = re.compile(r"<<step\s+([a-z][a-z0-9_]*)\((.*?)\)\s*->\s*(-?\d+(?:\.\d+)?)>>", re.S)
ARG_RE = re.compile(r"\s*([a-z][a-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*$")

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "half": 0.5, "twice": 2, "double": 2, "triple": 3, "dozen": 12, "once": 1, "single": 1, "pair": 2,
         "quarter": 4, "third": 3, "hundred": 100, "thousand": 1000, "week": 7, "weeks": 7, "year": 12,
         "hour": 60, "hours": 60, "day": 24, "days": 24}
CONSTANTS = [1, 2, 0.5, 7, 12, 24, 60, 100, 1000]  # unit-conversion constants, GSM8K only


def norm(x):
    """Canonical string for a number."""
    x = float(x)
    return str(int(x)) if x == int(x) else repr(round(x, 10))


def question_pool(question, constants=True):
    pool = set()
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", question):
        s = m.group(0).replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        pool.update([norm(v), norm(-v)])
    low = question.lower()
    if constants:
        for w, v in WORDS.items():
            if re.search(rf"\b{w}\b", low):
                pool.add(norm(v))
        pool.update(norm(c) for c in CONSTANTS)
    return pool


def parse_steps(text, q_pool, constants=True):
    pool = set(q_pool)
    rows, prov_ok = [], True
    for m in RECORD_RE.finditer(text):
        name, args, out = m.group(1), m.group(2), m.group(3)
        inputs = {}
        ok = True
        for a in args.split(","):
            am = ARG_RE.match(a)
            if not am:
                ok = False
                break
            inputs[am.group(1)] = am.group(2)
        if not ok:
            continue
        valid = all(norm(v) in pool for v in inputs.values())
        prov_ok &= valid
        rows.append({"name": name, "inputs": inputs, "output": out, "inputs_from_pool": valid})
        pool.add(norm(out))
    return rows, prov_ok


class PoolMask:

    def __init__(self, tok, vocab_size):
        self.V = vocab_size
        self.num = []     # (id, numeric prefix p, rest)
        others = []
        for i in range(len(tok)):
            s = tok.decode([i])
            m = re.match(r"[-0-9.]*", s)
            p = m.group(0)
            if p:
                self.num.append((i, p, s[len(p):]))
            elif s:
                others.append(i)
        self.others = torch.tensor(others, dtype=torch.long)

    def apply(self, rec_text, scores, pool):
        m = re.search(r"=([-0-9.]*)$", rec_text)
        if not m:
            return scores
        cur = m.group(1)
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool)
        for i, p, rest in self.num:
            c = cur + p
            if (rest == "" and any(v.startswith(c) for v in pool)) or (rest != "" and c in pool):
                allowed[i] = True
        if cur in pool:
            allowed[self.others] = True
        if not allowed.any():
            return scores
        return scores.masked_fill(~allowed.to(scores.device), -float("inf"))


class CraneDecoder:
    def __init__(self, model, tok, grammar_path):
        self.model, self.tok = model, tok
        self.proc = SyncodeLogitsProcessor(Grammar(grammar_path), tok, parse_output_only=True)
        self.pool_mask = PoolMask(tok, model.config.vocab_size)
        self.n_text_tokens = len(tok) - len(tok.added_tokens_decoder)
        self.digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        self.eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    @torch.no_grad()
    def generate(self, prompt_ids, q_pool, max_new=700, max_rec_tokens=120, prefix_ids=None, prefix_outputs=()):
        """prefix_ids / prefix_outputs: already-written reasoning (token ids) and its step outputs, used by the
        intervention test to continue generation after an edited step. Returns only the continuation."""
        dev = self.model.device
        pool = set(q_pool) | {norm(o) for o in prefix_outputs}
        prompt_ids = list(prompt_ids) + list(prefix_ids or [])
        ids = torch.tensor([prompt_ids], device=dev)
        out = self.model(input_ids=ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        gen, rec_start, stats = [], None, {"records": 0, "aborted": 0}
        for _ in range(max_new):
            scores = logits.float()
            if rec_start is not None:
                full = torch.tensor([prompt_ids + gen])
                scores = self.proc(full, scores)
                scores[:, self.n_text_tokens:] = -float("inf") 
                if not torch.isfinite(scores).any():
                    scores = torch.full_like(scores, -float("inf"))
                    scores[:, self.digit_ids] = logits.float()[:, self.digit_ids]
                scores = self.pool_mask.apply(self.tok.decode(gen[rec_start:]), scores, pool)
            nxt = int(scores.argmax(-1))
            if nxt in self.eos:
                break
            gen.append(nxt)
            if rec_start is None:
                if self.tok.decode(gen[-4:]).endswith(START):
                    rec_start = len(gen)
                    self.proc.reset()
            else:
                rec = self.tok.decode(gen[rec_start:])
                m = RECORD_RE.match(START + rec) if ">>" in rec else None
                if m:
                    pool.add(norm(m.group(3)))
                    stats["records"] += 1
                    rec_start = None
                elif len(gen) - rec_start > max_rec_tokens:
                    stats["aborted"] += 1
                    rec_start = None
            o = self.model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1]
        return self.tok.decode(gen), stats
