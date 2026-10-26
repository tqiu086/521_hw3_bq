# Auditable free-form reasoning with a CRANE-style decoder

Qwen2.5-7B-Instruct (4-bit nf4) writes free-text reasoning. When it writes the delimiter `<<step`, SynCode constrains
generation with a grammar until `>>`, producing a step record such as
`<<step total_apples(apples_per_box=12, boxes=5) -> 60>>`. The step records are parsed into a table and audited.
Tasks: 200 GSM8K test problems and 200 BBH `multistep_arithmetic_two` problems (sampled with seed 0).

## Code

| File | Purpose |
|---|---|
| `step_grammar.lark` | Lark grammar of one step record (free step/argument names, numeric inputs and output). |
| `decoder.py` | The decoder. `CraneDecoder` alternates free generation and SynCode-constrained generation. `PoolMask` limits argument values to numbers from the question, earlier step outputs, and a few constants. Also holds the step parser (`parse_steps`). |
| `run_eval.py` | Runs the methods on the problems: `crane` (my decoder), `cot` (unconstrained CoT), `template` (fixed SSRM-style step names in the prompt), `crane_free` (same prompt as `crane`, no constraints; control run). Writes `results/<task>_<method>.jsonl`. |
| `summarize.py` | Prints the table: parse rate, provenance rate, accuracy, steps per question, for each task and method. |
| `audits.py` | Faithfulness audits (S1 input provenance, S2 step arithmetic recomputed, S3 answer = last step; T1 typicality via name embeddings, k-means and a bigram model) with Fisher exact tests; also summarizes the intervention test. Writes `results/audit_table.md`. |
| `intervene.py` | Intervention test: edits the output of one step, continues generation from there, and checks whether the final answer changes. Writes `results/intervene_<task>.jsonl`. |

## Data and results

| Path | Contents |
|---|---|
| `data/gsm8k_test.jsonl` | GSM8K test set (from openai/grade-school-math). |
| `data/multistep_arithmetic_two.json` | BIG-Bench Hard task file (from suzgunmirac/BIG-Bench-Hard). |
| `results/<task>_<method>.jsonl` | One line per problem: output text, parsed step table, predicted/gold answer, `parsed`, `provenance_ok`, `correct`. |
| `results/intervene_<task>.jsonl` | One line per edited output: which step, old/new value, original/new answer, continuation. |
| `results/audit_table.md` | Audit table (% failing, accuracy when passing/failing, Fisher p) and the name clusters. |

## Other folders and logs

| Path | Contents |
|---|---|
| `models/qwen2.5-7b-instruct/` | Model weights (about 15 GB). |
| `cache/` | SynCode's cached token-mask store for the grammar (rebuilt automatically if deleted; about 25 s). |
| `run.log`, `run_free.log`, `intervene.log`, `dl.log` | Logs of the main run, the control run, the intervention run, and the model download. |

## Reproduce

Tested with Python 3.12, an NVIDIA GPU with 16 GB (the model runs in 4-bit, about 6-7 GB), and these versions:
torch 2.14 (CUDA 12.6), transformers 4.53.2, syncode 0.4.16, bitsandbytes 0.50.2, accelerate 1.15.0, huggingface_hub 0.36.2, numpy 2.5.

```bash
# 1. Environment
conda create -n ssrm python=3.12 -y
conda activate ssrm
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.53.2 accelerate bitsandbytes syncode==0.4.16 huggingface_hub numpy

# 2. Model (about 15 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/qwen2.5-7b-instruct', allow_patterns=['*.json','*.safetensors','*.txt'])"

# 3. Data (skip if data/ is already present)
mkdir -p data
curl -L -o data/multistep_arithmetic_two.json https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/multistep_arithmetic_two.json
curl -L -o data/gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

# 4. Experiments (run from this folder; each step resumes from existing result files)
python run_eval.py --n 200                        # crane, cot, template on both tasks
python run_eval.py --methods crane_free --n 200   # control: same prompt, no constraints
python intervene.py                               # intervention test (needs the crane results)
python audits.py                                  # audit table -> results/audit_table.md
python summarize.py                               # accuracy / parse table
```

On the first run SynCode builds a token-mask cache for the grammar (about 25 s) in `cache/`.
