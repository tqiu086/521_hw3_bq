## hw3 bonus question

### (a) Design

```
start:   "<<step " NAME "(" [args] ") -> " OUT ">>"
args:    arg (", " arg)*
arg:     NAME "=" IN
NAME:    /[a-z][a-z0-9_]{1,40}/  // choose step name and argument names by itself
OUT:     /-?\d+(\.\d+)?/         // result of the step (in number)
IN:      POOL
POOL:    "12" | "5" | "60" | ... // rebuilt before every step record
```

- The model picks NAME (the step name), the argument names, and the number of args itself. Any number of step records can appear in this text, one after another.
- IN is a closed set pool. The pool starts when all numbers extracted from the question (e.g. 12, 5). Each time a record closes, its OUT is added to the pool (e.g. 60).
- OUT is any number. The grammar can’t check that it is arithmetically right, so I do it later.
- Final answer is an optional second record, `<<answer IN>>`, ties the final answer to the pool

**How it switch mode**:

- When LLM writes free reasoning text with no masking, it's unconstrained mode.
- When the decoder watches the output string, see the start delimiter `<<step`, switch to constrained mode. In this mode, the delimiter is legal.
- When the parser accepts `>>`, the record is complete. The coder parses the record, adds out to the pool and switches back to unconstrained mode, to make the model can reason again, plan the next step or write records. 
- When finih, it ends generating, then write the `<<answer ... >>` record. 

**Differences**:

- For step names, SSRM has a fix set task, I make it accept any idedntifier and the LLM an modify it. 
- For number of sets, SSRM set its own template but I also make LLM choose freely.
- For format adherence, SSRM needs fine tune, I don't do it but make it enforced by the grammar at decode time. 
- For reasoning text, SSRM make the entire CoT as the steps, I use free text plus structured records. 
- For audits, SSRM focuses on task-specific structured rules (`get_data`), I use task-agnostic audits to make sure no unused steps (provenance, arithmetic re-check, answer equals last output).

### (b) Implementation

Choose Qwen 2.5-7B-Instruct, loadded in 4-bit (nf4) (just because 7b is not compactable with my laptop). I decode it greedily. Evaluate on 200 randdomly sampled GSM8k test problems and 200 BIG-BENCH HARD `multistep_arithmetic_two` problems. 

**Decoder**:

- Model writes free text with no restriction.
- When text ends with th start delimiter, the decoder switches to constrained mode. SynCode computes, from Lark grammer, which takens comes next. Every otehr token gets a logit of -$\infty$
- I wrote a second mask applies at argument-value positions. It only lets the model write numbers that appear in the question, which are outputs of ealier steps, or are a small set of conversion constants. Number words in question also count ("twice", "half").
- When record closes with `>>`, its output is added to the followed pool and the decoder goes back to free text. 
- Records that run past 120 tokens are abandoned, decoder returns to free text. 

**Parsing**:

I extract all records from an output with regex into a table of step name, inputs, putput. An output counts as "parsed" if it has at least one record and a final answer. For the template base line, parsing also requires thaat every step name is one of the templats's name.

**Baselines**:

- CoT is not constrained
- The prompt lists the only allowed step names, decoded without constraints.
- Because without constraints, I separate the effect of the prompt from the effect of the constraint. 

**Results**:

| Task  | Method            | Parsed into a step table | Accuracy |
|-------|-------------------|--------------------------|----------|
| GSM8K | My decoder        | 85 %                     | 75 %     |
| GSM8K | Unconstrained CoT | 0% (no step records)     | 90.5 %   |
| GSM8K | Fixedd template   | 99 %                     | 59 %     |
| BBH   | My decoder        | 100 %                    | 36 %     |
| BBH   | Unconstrained CoT | 0 %                      | 83.5 %   |
| BBH   | Fix template      | 100 %                    | 6 %      |

- The 15% of GSM8K outputs that didn’t parse are cases where the model never wrote a step record and just did free-text algebra.
- My decoder is more accurate than the fixed template on both tasks, but less accurate than plain CoT. The control run suggests the constraint itself costs about 14 points on GSM8K and about 6.5 on BBH.
- BBH here is the numeric arithmetic task, which is similar in type to GSM8K.

### (c) faithfultest

I run three kinds of test on my decoder’s outputs (200 GSM8K and 200 BBH problems). For a structured or typicality audit, an output *fails* if it breaks the rule. For each audit I report the percentage failing, the accuracy among passing and failing outputs, and the p-value of Fisher’s exact test (two-sided, computed on the 2 by 2 table of pass/fail $\times$ correct/incorrect). Outputs with no step table count as failing the structured audits, because they can’t be checked.

**(i) Structured audits.** Rules over the table of steps, where each input is a number from the question or an earlier step output.

- S1, input provenance: every input of every step is defined in the question (plus the allowed constants) or is the output of an earlier step.
- S2, step recomputed: I recompute each step in Python. Since step names are free, I check that the output can be produced from the step’s inputs by arithmetic. For 1–3 inputs I accept any expression over them (any order, any of + − $\times$ $\div$). For 4 or more I try the inputs in the written order with any operators.
- S3, answer = last step: the final answer in the text equals the output of the last step.

**(ii) Typicality audit.** The model picks the step names, so `ads_total` and `total_ads` can be the same operation. To make names comparable:

- I collect every unique step name across all outputs.
- I embbed each name using Qwen’s hidden state (layer 20, last token of a short prompt "A step in a math solution is named: ...").
- I cluster the embeddings into 6 clusters with k-means (cosine similarity).
Each output becomes a sequence of cluster IDs with start and end markers.
- I fit a bi-gram model with add-one smoothing over these sequences. Each output is scored by its average log-probability per transition, computed leave-one-out, so an output doesn’t see its own counts.
- An output fails if it is in the lowest $20%$ of scores for its task and method. The $20%$ cutoff is my choice.

**(iii) Intervention test.** For each output with at least one step, I pick one step at random and change its output by a random amount ($\pm 2$ to $\pm 9$, or $\pm 20$ to $\pm 90$). I keep the text up to the end of the edited record and let the decoder continue from there, with the edited value added to the allowed pool. The output “fails” if the final answer is unchanged, which means the answer didn’t depend on that step.

| Task  | Test                  | N   | Percentage fail | Acc if pass | Acc if fail |
|-------|-----------------------|-----|-----------------|-------------|-------------|
| GSM8K | S1 provenance         | 200 | 15              | 82.4        | 33.3        |
| GSM8K | S2 recomputed         | 200 | 38              | 83.9        | 60.5        |
| GSM8K | S3 answer = last step | 200 | 23.5            | 88.9        | 29.8        |
| GSM8K | Typicality            | 170 | 20              | 80.9        | 88.2        |
| GSM8K | Intervention          | 170 | 38.8            | 81.7        | 83.3        |
| BBH   | S1 provenance         | 200 | 0               | 36.0        | n/a         |
| BBH   | S2 recomputed         | 200 | 14.5            | 30.3        | 10.3        |
| BBH   | S3 answer = last step | 200 | 0.5             | 36.2        | 0           |
| BBH   | Typicality            | 200 | 13              | 38.5        | 19.2        |
| BBH   | Intervention          | 200 | 6               | 35.1        | 50          |

- Structured audits work. Outputs that fail them are much less accurate. S3 on GSM8K is the strongest ($88.9%$ vs $29.8%$, $p=1e-14$), and S2 on BBH is significant ($p=0.0014$). S1 on BBH and S3 on BBH almost never fail, so they say nothing there.
- Typicality is weak. It's not significant on GSM8K ($p=0.45$) and borderline on BBH ($p=0.078$). Names are mostly specific to the problem (`amy_age`), so clusters don’t line up with operations.
- Intervention shows unfaithfulness on GSM8K. In $38.8%$ of outputs the edited step did not change the answer, for example when the model noticed the edit and wrote the correct value back. On BBH almost all answers depend on the steps ($6.0%$ unchanged).
- Limits are that the $20%$ typicality cutoff is arbitrary; my arithmetic audit is lenient, so some wrong steps pass; and a random step may legitimately not affect the answer.

### (d) Analysis

Numbers come from 200 GSM8K and 200 BBH `multistep_arithmetic_two problems`. 

| Method                                   | GSM8K acc | BBH acc | Inputs from question or earlier steps (GSM8K / BBH) |
|------------------------------------------|-----------|---------|-----------------------------------------------------|
| Unconstrained CoT                        | 90.5      | 83.5    | n/a                                                 |
| Fixed Template                           | 59        | 6       | 54/87                                               |
| Same prompt unconstrained (`crane_free`) | 89        | 42.5    | 73/90                                               |
| My decoder (constrained)                 | 75        | 36      | 85/100       

**(i) The free structrue is easier to audit**

- Plain CoT has nothing to audit. It writes no step records, so all structured audits fail by default ($0%$ parsed). The template prompt produces records but follows the format loosely, so its provenance audit fails on $46%$ of GSM8K outputs. In my decoder the format is guaranteed: 100% of BBH outputs and $85%$ of GSM8K outputs parse.
- My decode's constraint removes one whole class of failure. Unconstrained generation with the same prompt breaks input provenance in $27%$ of GSM8K outputs and $10%$ of BBH outputs. The constraint brings this to zero among outputs that have records.
- The audits carry information. On GSM8K, outputs that fail the “answer equals last step” audit are $29.8%$ accurate vs $88.9%$ for those that pass (Fisher $p=1e-14$). The audit for step arithmetic recomputed from its inputs also separates them (BBH: $10.3%$ vs $40.4%$, $p=0.0014$).
- Free names weaken some audits.
-- Step names are mostly entity-specific (`amy_age`, `ankanis_fish`). My 6 embedding clusters therefore don't mean "add" or "multiply", and the typicality audit adds little (GSM8K $p=0.45$, BBH $p=0.078$).
-- With a fixed template the step vocabulary is known, so a typicality model is more meaningful. Free names trade that away.
- Audits can be fooled. My arithmetic audit tolerates up to three-operand expressions in any order, so a wrong step like `-9 - -7 + 2 + 2 -> -4` (true value $2$) passes. On BBH, $101$ of the $128$ wrong answers pass every structured audit.


**(ii) Constrained generation lowers the accuracy**

Against plain CoT my decoder is 15.5 points lower on GSM8K ($75.0$ vs $90.5$) and $47.5$ points lower on BBH ($36.0$ vs $83.5$). The control run splits that gap into two sources.

- The step-record prompt costs accuracy on BBH, not on GSM8K. Prompt alone, with no constraints, gives GSM8K $89.0$ (almost the same as CoT) and BBH $42.5$ (far below CoT at 83.5). On BBH the model puts several operations into one record and mis-evaluates it, instead of writing out each step.
- The constraint itself costs a further 14.0 points on GSM8K ($89.0$ to $75.0$) and $6.5$ on BBH ($42.5$ to $36.0$). Paired by problem, $35$ GSM8K problems are solved only without constraints, against $7$ only with them (on BBH, $33$ vs $20$).
- What I observed behind the GSM8K loss:
    - $20$ of the $50$ wrong answers have no step records at all, so the model never entered constrained mode and did free algebra.
    - The rest are mostly a restricted input set. The pool limits argument values to numbers from the question, earlier outputs, and a few constants. If the model needs a value outside that set, it has to pick a different, wrong one.
- The trade-off is that the constraint buys provenance ($100%$ vs $90%$ on BBH, $85%$ vs $73%$ among GSM8K outputs) at the cost of accuracy.

**(iii) The model can write correct step records but calculate the answer in the free text**

My tests find it in $23.5%$ of GSM8K outputs (the final answer differs from the last step’s output).

**Caught by my tests**:

- **(GSM8K, gold 90)**. The step records correctly give $20 − 55 = −35$. The free text then declares “the result is 0 because she cannot make the purchase” and answers 0. The answer does not follow from the table.
- **(GSM8K, gold 4)**. The answer comes from algebra written in the free text, not from any step. It is correct, but the steps don't support it.
- Intervention -- I edited the step "second parenthesis" from $-13$ to $17$. The model noticed in free text (`"-9 - 3 equals -12 … equals -13"`), rewrote the step and kept the answer 30. The answer did not depend on the edited step, so the table is not what produces the answer.
    - Overall the answer survived an edit in $38.8%$ of GSM8K outputs and $6.0%$ of BBH outputs.

**Missed by my tests**:

- **(GSM8K, gold 245)**. The steps are internally consistent and end at $24500$ cents. The free text then converts to $245$ dollars. The answer follows from the steps and passes every audit, but the unit conversion is only in the free text.
- **(GSM8K, gold 300)**. Every record recomputes correctly from its inputs. The final steps compute the monitors’ total ($600$), divide by $2$, then divide again and answer 150. This passes all structured audits and changes when I edit a step, but is wrong.
- BBH arithmetic -- `((-9 - -7 + 2 + 2) + …)`: a step writes `-> -4` where the true value is $2$. The lenient arithmetic audit accepts it, and the answer is wrong ($-19$, gold $-13$).