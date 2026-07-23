# Lean: Case Study

## 1. The problem

LLMs are frequently verbose by default: they restate the question, hedge
("Let's think step by step..."), and pad reasoning with words that don't
change the answer. Every extra token costs money (API pricing is per-token)
and latency (more tokens to generate = slower response). If a model can be
taught to give the *same correct answer* in fewer tokens, that's a direct,
measurable efficiency win — cheaper and faster, with no loss of quality.

Lean's goal: fine-tune a small open-weight model to be more concise while
staying exactly as correct, on a narrow, well-defined task (grade-school math
word problems, GSM8K), and prove the gain with a mechanical measurement
rather than a vibe.

## 2. The technique: compression-distillation

This is not generic fine-tuning — it's a specific recipe:

1. Take a **verbose gold answer** for a task (in our case, GSM8K's own
   dataset answers, which show full step-by-step arithmetic reasoning).
2. Have a **teacher** rewrite it as short as possible while keeping it
   fully correct.
3. **Mechanically verify** the shortened version still arrives at the right
   final answer — never trust the teacher's rewrite blindly, because an LLM
   (or a script) can silently drop a needed step and get the wrong number.
4. Keep only the pairs that pass verification. Discard/log the rest as the
   **rejection rate** — the single most important honesty metric in this
   whole pipeline. A 0% rejection rate on trivial compression is expected;
   a 0% rate on aggressive compression would be suspicious.
5. Fine-tune the base model on the surviving (prompt, short-completion)
   pairs using supervised fine-tuning (SFT).

The "teacher" in step 2 is normally a stronger LLM (e.g. via API). We
deliberately did **not** use the Anthropic API here — no API key, no paid
usage credits, by explicit choice — see [[feedback_no_api_credits]]. Instead:

- For the first 20-example smoke test, Claude (this session, running under
  the Pro plan) acted as the teacher directly, hand-writing the compressed
  answers and having them checked mechanically.
- For the full 7,473-example run, we used a **deterministic script** instead
  of a per-example LLM call. This was possible because GSM8K's gold answers
  already contain very regular scaffolding: line-by-line arithmetic with
  inline calculator annotations like `<<48/2=24>>`. A regex strips the
  annotations and joins the remaining lines. This is a legitimate mechanical
  "compression" pass, verified the same way (final-number match), but it's
  worth being honest about what it actually did (see caveat in §4).

## 3. Tools, and why each one

| Tool | Role | Why this one |
|---|---|---|
| **LoRA** (Low-Rank Adaptation) | Fine-tuning method | Trains small rank-decomposed adapter matrices instead of all model weights. A 1.5B model's full fine-tune needs far more VRAM than a free-tier GPU provides; LoRA needs a fraction of that and produces a small (tens of MB) adapter file instead of a full model copy. |
| **4-bit quantization** (bitsandbytes) | Memory reduction | Loads the base model's weights compressed to 4 bits during training, which is what makes a 1.5B model fit and train on a 16GB T4. |
| **Unsloth** | Training accelerator | Fuses LoRA + 4-bit loading and rewrites the training kernels for speed — roughly 2x faster and lower memory than plain HuggingFace PEFT for an equivalent LoRA run, at no added code complexity. |
| **TRL's `SFTTrainer`** | Training loop | Standard, maintained supervised fine-tuning loop (loss on next-token prediction over prompt+completion text) — no reason to hand-roll one. |
| **PEFT** | Underlying adapter library | What TRL/Unsloth build LoRA support on top of. |
| **Kaggle free GPU** (Tesla T4, dual-GPU instance) | Compute | Free 30-hour/week GPU quota, no local GPU required. |
| **Hugging Face `datasets`** | Data loading | Pulled the full GSM8K train split (7,473 rows) directly, and loads `pairs.jsonl` into the training pipeline. |
| **A regex-based mechanical compressor** (this project, not a generic tool) | Teacher stand-in | See §2 — replaces an LLM teacher call for this specific, already-regular dataset format. |
| **`eval.py`** (next step, not yet run) | Measurement | Compares the fine-tuned adapter against the base model on accuracy (via the same final-number check) and mean output token length on a held-out set. |

## 4. The data, honestly

- **Source:** full GSM8K train split, 7,473 question/answer pairs, pulled
  live via `datasets.load_dataset("openai/gsm8k", "main", split="train")`.
- **Compression applied:** strip `<<...>>` calculator annotations, join
  reasoning lines with `; `, keep the `#### <answer>` marker.
- **Verification:** for every pair, extract the last number in both the
  compressed and original text and require they match. 7,473 kept, 0
  rejected.
- **Important caveat, stated plainly:** measuring average word count
  before/after showed **~0% reduction**. GSM8K's gold answers are already
  terse — the annotations we stripped are inline symbols, not filler prose.
  This means the *data prep step* did not teach much brevity by itself.
  The real test of whether the model becomes more efficient happens in
  `eval.py`, comparing the **fine-tuned model's own generated outputs**
  against the **base model's own generated outputs** — base models tend to
  restate the question and narrate more than the dataset's gold answers do,
  so that's where a real compression gap, if any, will show up.
- **Scope decision:** this data is 100% grade-school math word problems.
  We explicitly chose (see conversation) to keep this GSM8K-only for v1
  rather than mixing in other task types. That means: whatever efficiency
  gain this produces will very likely be narrow to this task shape, not a
  general "this model is now more concise at everything" result. Extending
  to general-purpose efficiency would require repeating this same pipeline
  on other task types and combining the data.

## 5. The training run

Configuration (`train_kaggle.ipynb`):

- **Base model:** `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (1.56B params, 4-bit)
- **LoRA:** rank 16, alpha 16, dropout 0, applied to all attention + MLP
  projection layers (`q/k/v/o_proj`, `gate/up/down_proj`)
- **Trainable parameters:** 18,464,768 of 1,562,179,072 — **1.18%** of the
  model. This is the whole point of LoRA: 98.8% of the weights are frozen.
- **Batch size:** 2 per device × 8 gradient accumulation steps = effective
  batch of 16 (only 1 GPU actually used out of the 2 available — Unsloth
  reported "Data Parallel GPUs = 1")
- **Epochs:** 2 over 7,473 examples → 936 total optimizer steps
- **Learning rate:** 2e-4
- **Max sequence length:** 1024 tokens

**Full run result:** completed all 936 steps in 53:16 (3,209s), 4.66
samples/sec. Final `train_loss = 0.468`.

Loss trajectory (step: loss):
- Step 10: 1.292 → step 50: 0.542 — the steep initial drop, model rapidly
  fitting the compressed-answer format.
- Step 100–450: loss plateaus and oscillates in a **0.44–0.54** band. This
  is normal — most of epoch 1's easy gains are exhausted, and the model is
  now fine-tuning within noise of the optimizer/batch variance rather than
  making large jumps.
- Step 460 onward (into epoch 2): a second, smaller step down to a
  **0.37–0.43** band, bottoming near step 630 (0.357, the run's lowest single
  point) and step 890 (0.375). This second dip is the second epoch seeing
  the same 7,473 examples again and fitting them slightly tighter.
- No divergence, no NaN, no loss spike at any point — a clean, textbook LoRA
  SFT curve for a small dataset run twice.

**What this loss curve does and doesn't tell you:** it confirms the model
learned to reproduce the compressed-answer *style* (line-per-step arithmetic
ending in `#### N`) reliably. It says nothing about whether the model is
correct or concise on questions it wasn't trained on — that's exclusively
what §6's evaluation step measures. A model can drive this loss to near-zero
by memorizing training examples while still failing badly on held-out ones.

**Two real bugs hit and fixed along the way** (both are useful to understand,
not just "it broke"):

1. **Dataset path mismatch.** The notebook assumed Kaggle would mount an
   uploaded dataset at `/kaggle/input/<dataset-name>/`. In practice, Kaggle
   nested it one level deeper: `/kaggle/input/datasets/<your-username>/lean-pairs/pairs.jsonl`.
   Diagnosed by running a small `os.walk` snippet to print every file under
   `/kaggle/input` rather than guessing.
2. **TRL API change.** `SFTConfig` in the TRL version installed on this
   Kaggle image had renamed `max_seq_length` to `max_length` — an upstream
   library change unrelated to our code, caught by reading the `TypeError`
   traceback and fixing the one keyword argument.

Both are the kind of environment drift you should expect any time a fresh
cloud notebook installs "whatever's latest" rather than pinned versions —
not a flaw in the plan itself.

**Output cleanup:** Kaggle's download produced a duplicate
`adapter_model (1).safetensors` (browser re-download artifact, byte-identical
to `adapter_model.safetensors`) alongside the real adapter files in
`runs/lean-v1/adapter/`. Deleted — harmless, but worth knowing your browser
can silently double-save large files during a folder download.

## 6. Evaluation: what broke and what worked

`eval.py` compares the fine-tuned adapter against the base model on a
**held-out set** — 15 questions from GSM8K's *test* split (`data/eval.jsonl`),
which is disjoint from the 7,473 *train*-split rows the model trained on.
This is the step that actually tells you whether training helped, since the
training loss curve (§5) only measures fit to data the model has already seen.

**Two real bugs found and fixed before this could even run:**

1. **`eval.py` assumed the adapter folder was a complete model.** It called
   `AutoModelForCausalLM.from_pretrained(args.model)` directly on
   `runs/lean-v1/adapter/`, but a LoRA adapter directory only contains the
   small delta weights (`adapter_model.safetensors`, ~74MB) and
   `adapter_config.json` — not the full 1.5B base model. Loading it as if it
   were a standalone model would either fail outright or silently produce
   garbage. **Fix:** load the base model separately, then apply the adapter
   on top via `peft.PeftModel.from_pretrained(base, adapter_path)`.
2. **The adapter's recorded base model is 4-bit quantized
   (`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`), which needs `bitsandbytes` +
   GPU support.** This machine's PyTorch is CPU-only
   (`torch==2.12.0+cpu`, `torch.cuda.is_available() == False`), and
   bitsandbytes 4-bit inference isn't practically usable on CPU. **Fix:**
   load the plain full-precision base (`Qwen/Qwen2.5-1.5B-Instruct`) instead.
   This works because LoRA adapters are just additive low-rank deltas on top
   of the same linear layers — they don't depend on the base being quantized,
   so a full-precision base + the same adapter weights is a valid combination,
   just slower per-token than the quantized version would be on a GPU.

**Why the eval sample size matters:** this machine has no GPU, so generating
up to 512 tokens per example, twice (fine-tuned + base, for a fair
comparison), on CPU is slow. We ran two passes: an initial n=15 smoke test,
then a n=100 confirmation run once the first result looked like a real
regression worth checking rather than trusting on 15 examples.

**Results, n=15 (initial, smoke-test scale):**

| | Base model | Fine-tuned (Lean adapter) |
|---|---|---|
| Accuracy | 60.0% | 40.0% |
| Mean output tokens | 313.3 | 93.6 |

**Results, n=100 (confirmation run, held-out GSM8K test questions):**

| | Base model | Fine-tuned (Lean adapter) |
|---|---|---|
| Accuracy | 69.0% | 51.0% |
| Mean output tokens | 299.4 | 93.8 |

The larger sample **confirms the pattern, with a narrower gap than the n=15
run suggested**: the 20-point accuracy drop at n=15 was partly small-sample
noise (base model's true rate is closer to 69%, not the 60% the small
sample showed) — but an **18-point real accuracy drop (69%→51%) persists**
at n=100, alongside a consistent **~69% token reduction** (299→94) in both
runs. This is exactly why the n=15 result was flagged as "suggestive, not
conclusive" rather than reported as final — the direction held, the
magnitude moved.

**Interpretation — and a correction to an earlier hypothesis.**

The first version of this case study guessed the accuracy drop was caused by
the mechanical compressor discarding needed reasoning steps during data prep.
**That hypothesis doesn't survive a closer check of the actual training
data.** Measuring the training completions directly:

```
avg words per training completion: 51.7
```

That's essentially unchanged from the original GSM8K gold answers (§4 already
measured ~0% word reduction from the compression step — the regex only
stripped inline `<<...>>` calculator annotations, never removed a reasoning
line). **The training data itself was barely compressed.** Yet the fine-tuned
model generates dramatically shorter outputs at inference (93.8 tokens) than
the base model (299.4 tokens) — a much larger gap than anything present in
the training labels. So the cause isn't "the compressor deleted a step the
model needed." Something else is going on:

1. **The compression effect is real and stable.** ~70% fewer tokens per
   answer, consistently, across both sample sizes (n=15 and n=100). LoRA
   reliably taught the terse `step; step; #### N` format — and generalized
   that terseness *beyond* what the training labels actually modeled.
2. **The accuracy cost is also real, not sampling noise.** An 18-point gap
   on 100 examples (69 vs 51 correct) is a large, structural effect.
3. **The likely real cause: less inference-time "thinking."** GSM8K-style
   math accuracy in LLMs is known to correlate with how much intermediate
   token-by-token reasoning a model performs before committing to an answer
   (the same principle behind chain-of-thought prompting). By training the
   model to imitate compact, minimal-looking answers, we didn't just teach
   it a *formatting* preference — we taught it to allocate fewer tokens to
   reasoning at generation time, even on problems where it would benefit
   from working through more steps. The label text barely changed; the
   model's *inference behavior* changed a lot. That's a genuinely more
   interesting finding than a data-prep bug: it's empirical evidence of an
   accuracy/verbosity trade-off in chain-of-thought reasoning, reproduced
   with a real fine-tune rather than cited from a paper.

**What this means for the project, honestly:** the pipeline works
end-to-end and produces a real, confirmed compression effect, but the
adapter sits at one point on an accuracy/efficiency trade-off curve — not a
bug to patch, but a curve worth mapping. The most informative next move
available without retraining from a different data recipe was to isolate
*why* it happens — which we then ran as a real ablation (§6.1).

## 6.1 Ablation: is it the line-collapsing format, or something deeper?

**Hypothesis tested:** v1's compressor joined every reasoning line into one
semicolon-separated line (`step; step; #### N`) instead of the GSM8K
original's natural multi-line layout. Maybe *that* structural collapse — not
the SFT process itself — is what taught the model to "answer in one terse
burst" and skip reasoning depth on harder, unseen questions.

**Design:** built a second dataset, `pairs_v2.jsonl`, using the mildest
possible edit — strip only the `<<...>>` calculator annotations, keep the
**original multi-line structure** (`\n` between steps, exactly like GSM8K's
own gold answers). Verified identical word count to v1 (51.7 avg words,
same as before) — the *only* variable changed between v1 and v2 is
line-format, nothing else. Trained a second LoRA adapter (`lean-v2`) on this
data, same config, same 7,473 examples, same 2 epochs.

**Result, same 100-example held-out set:**

| | Base model | v1 (line-collapsed) | v2 (structure-preserved) |
|---|---|---|---|
| Accuracy | 69.0% | 51.0% | 50.0% |
| Mean output tokens | 299.4 | 93.8 | 89.7 |

**v1 and v2 are statistically indistinguishable** (51.0% vs 50.0% accuracy,
93.8 vs 89.7 tokens) despite training on data that differs *only* in
whether reasoning steps are joined with `; ` or kept on separate lines.
**This falsifies the format-collapse hypothesis.** Preserving the original,
natural multi-line structure of the training labels made no measurable
difference to either accuracy or output length.

**What this actually shows:** the accuracy/brevity trade-off isn't caused by
*how* the compressed answers are formatted — it's an effect of **fitting the
model via SFT to reproduce GSM8K's already-terse gold-answer style at all**,
regardless of line-joining choices. The base model's own un-tuned behavior
(chatty, ~300 tokens, restating the problem, narrating in full sentences) is
what's being trained away; once the model is fit to *any* version of the
terse gold-answer style, it generalizes that terseness into reduced
reasoning depth on new questions, and formatting details around that
terseness don't move the needle. This is a cleaner, more specific finding
than the original hypothesis — and it's a real negative result, arrived at
by testing an idea and discarding it when the data didn't support it, not by
picking whichever explanation sounded best.

**One candidate next experiment was training dosage — tested next (§6.2)
rather than left as a suggestion.**

## 6.2 Ablation: does less training dosage trade back some accuracy?

**Hypothesis tested:** v1 and v2 both trained 2 full epochs to a loss
plateau (§5). Maybe that's over-fitting the terse style harder than
necessary — 1 epoch might preserve more of the base model's reasoning depth
while still cutting some tokens.

**Design:** trained a third adapter (`lean-v3`) on the exact same data as
v2 (structure-preserved), same LoRA config, but **1 epoch instead of 2** —
isolating training dosage as the only variable.

**Result, same 100-example held-out set:**

| | Base | v1 (2 epochs, collapsed) | v2 (2 epochs, structured) | v3 (1 epoch, structured) |
|---|---|---|---|---|
| Accuracy | 69.0% | 51.0% | 50.0% | **45.0%** |
| Mean output tokens | 299.4 | 93.8 | 89.7 | **86.4** |

**This is the opposite of the hypothesis.** Less training didn't trade back
accuracy for less compression — v3 is *worse on both axes* than v1/v2: lower
accuracy (45% vs ~50%) *and* fewer tokens (86.4 vs ~90-94). If anything, 1
epoch looks like a strictly worse operating point, not a gentler one.

**What this suggests:** the model isn't gradually and smoothly trading
accuracy for brevity as training progresses — a partially-trained (1 epoch)
adapter seems to pick up the terse *style* about as fast as the fully
trained one (tokens are already low at 1 epoch), but hasn't yet had the
extra epoch's worth of gradient steps needed to *consolidate reliable
arithmetic* under that new style. In other words: the format change happens
fast; recovering reasoning reliability under the new format (to the extent
2 epochs partially does) takes longer. This reframes the earlier "trade-off
curve" framing — the data here doesn't support a smooth dial at all; it
looks more like the style shift and the reasoning-quality recovery move on
different timescales, and 1 epoch catches the model in a worse spot on both.

This too is reported as found, not adjusted after the fact — the hypothesis
was reasonable going in, and the result contradicts it, which is itself the
finding.

## 8. Revised honest summary

Lean is a working, end-to-end compression-distillation LoRA pipeline that
reliably teaches a small model to generate far fewer tokens per answer
(consistently ~70-71% fewer across three independent fine-tunes), at a real,
confirmed accuracy cost. Three hypotheses for the cause/shape of that cost
were tested in sequence, and evidence — not intuition — decided each one:

1. **Not explained by the data-prep step deleting needed reasoning** —
   training labels were barely shortened from GSM8K's originals (§4, §6).
2. **Not explained by reasoning-line-joining format** — v1 (collapsed) and
   v2 (structure-preserved) landed statistically indistinguishable (§6.1).
3. **Not a smooth, less-training-preserves-more-accuracy dial** — v3 (1
   epoch) did *worse* on both accuracy and tokens than v1/v2 (2 epochs),
   contradicting that hypothesis directly (§6.2).

The best-supported remaining explanation is that SFT on any terse-style
GSM8K answer teaches the model to generalize "answer tersely" into "reason
less" at inference time, that this style shift happens quickly during
training, and that recovering reliable arithmetic under the new style (what
the extra epoch in v1/v2 appears to partially provide) is a separate,
slower process than adopting the style itself.

## 9. v4 — self-distillation instead of gold-label compression

v1-v3 all shared one thing: the training target was GSM8K's own gold-label
style, uniformly ~52 words regardless of problem difficulty. §6-6.2
diagnosed *why* that hurt accuracy — training toward a fixed-length terse
target teaches the model to always answer briefly, including on harder
problems that need more reasoning. v4 tests the fix that diagnosis implies:
change what the model trains on, not just how it's trained.

**Design:**
1. **Generate the base model's own reasoning** on all 7,473 training
   questions (batched greedy decoding on a free Kaggle T4 — see
   `distill_generate.ipynb`).
2. **Keep only verified-correct output** — 5,780 of 7,473 (77.3%) matched
   the gold final answer. Self-distillation on the model's own correct
   behavior only; never trained on its own mistakes. Length already varied
   naturally with difficulty here: median 165 words, range 15-311 — nothing
   like v1-v3's flat ~52-word target.
3. **Trim boilerplate only, not reasoning.** Every generation had the same
   two filler patterns: an intro paragraph ("To determine X, we need to
   follow these steps:") and a closing restatement ("Thus, the answer is
   \boxed{N}."). A rule-based trim (`data/build_v4.py`) cut both, replacing
   the closing with a plain `#### N`, and left the actual derivation
   (equations, intermediate results) completely untouched. Result: 26.5%
   shorter, **0% rejection** — the trim never broke a correct answer,
   because it never touched the math.
4. Trained a fourth LoRA adapter (`lean-v4`) on this 5,780-example set, same
   config as v1/v2 (LoRA r=16, 2 epochs).

**Result, same 100-example held-out set:**

| | Base | v1 | v2 | v3 | **v4 (self-distill)** |
|---|---|---|---|---|---|
| Accuracy | 69.0% | 51.0% | 50.0% | 45.0% | **64.0%** |
| Mean output tokens | 299.4 | 93.8 | 89.7 | 86.4 | **248.4** |

**This is the result the earlier diagnosis predicted.** Accuracy dropped
only 5 points (69%→64%) instead of the 18-24 point drops v1-v3 all showed —
training on the model's own verified-correct, difficulty-scaled reasoning
preserves accuracy far better than compressing to a fixed terse template.
The honest other half: token reduction is far more modest too — 17% vs.
v1-v3's ~68-71%. That's not a shortcoming to explain away; it's the real
shape of the trade-off. Most of the token cost in the base model's answers
turns out to live in the *reasoning itself* — once you stop cutting
reasoning (to protect accuracy), you naturally stop cutting most of the
tokens too. v1-v3's large token savings came specifically from cutting the
part that also cost accuracy.

**What this means:** there isn't a free lunch where you keep 69% accuracy
*and* get to 90 tokens — v1-v3 already showed what happens when you push for
that (44-51% accuracy). v4 shows the other real point on the same frontier:
protect accuracy, and the achievable compression is real but modest (~17%,
from cutting only genuine boilerplate). Both are legitimate, honestly
measured points on the same accuracy/efficiency curve — which one to pick
depends on whether the application can tolerate a 5-point accuracy dip for
20% savings, or a 20-24 point dip for 70% savings.

## 10. Revised honest summary

Lean is a working, end-to-end fine-tuning pipeline that produced four
independent LoRA adapters, each testing a specific hypothesis about how to
trade tokens for accuracy on GSM8K math reasoning, with two hypotheses
falsified by evidence and one recipe (v4) that substantially improved the
trade-off once the diagnosis pointed at the actual cause:

1. **v1**: compress GSM8K's gold labels to a fixed terse template →
   70% fewer tokens, 18-point accuracy cost.
2. **v2**: isolate formatting as a variable → **falsified**, formatting
   doesn't matter, ruling out "line-collapsing caused the drop."
3. **v3**: isolate training dosage as a variable → **falsified in the
   opposite direction**, less training made both axes worse, ruling out a
   smooth accuracy/brevity dial.
4. **v4**: change the training *target* itself — self-distill on the base
   model's own verified-correct, difficulty-scaled reasoning, trimming only
   genuine boilerplate → 64% accuracy (vs. 69% base) at 248 tokens (vs. 299
   base), by far the best accuracy-preserving result of the four.

The throughline: the earlier diagnosis (§6-6.2) that SFT on a fixed-length
terse target teaches "reason less" wasn't just an explanation after the
fact — it made a testable prediction (train on adaptive-length,
verified-correct reasoning instead, and accuracy should recover) that v4
then confirmed. That is the actual deliverable here: not a single
efficiency number, but a diagnosed, predictive account of *why* naive
compression-distillation costs accuracy, and a concrete recipe that
recovers most of it.