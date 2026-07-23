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

**Remaining, better-targeted next experiments** (not run here, but now
correctly scoped by this ablation):

- **Vary how much the target answer diverges from base-model-style
  verbosity itself** — e.g. train on partially-shortened labels (say, keep
  one explanatory sentence per step rather than none) rather than varying
  formatting. This is the lever the ablation shows actually matters.
- **Fewer epochs or lower learning rate** — reduce how strongly the model
  commits to the terse style, since 2 full epochs to a loss plateau (§5) may
  be over-fitting a stylistic pattern harder than necessary.
- **An inference-time minimum reasoning-token floor** — a cheap mitigation
  that doesn't require retraining at all.

## 8. Revised honest summary

Lean is a working, end-to-end compression-distillation LoRA pipeline that
reliably teaches a small model to generate ~70% fewer tokens per answer —
confirmed across two eval sample sizes (n=15 and n=100) and two independent
training runs (v1, v2) — at a real, confirmed accuracy cost (69%→~50% at
n=100). Two hypotheses for the cause were tested in sequence and one was
ruled out by evidence rather than assumed: it is **not** explained by the
data-prep step deleting needed reasoning (labels were barely shortened), and
it is **not** explained by the reasoning-line-joining format (the v1/v2
ablation showed identical results despite that formatting difference). The
best-supported explanation is that SFT on any terse-style GSM8K answer
teaches the model to generalize "answer tersely" into "reason less" at
inference time — a specific, testable instance of the known link between
chain-of-thought length and accuracy on math reasoning tasks, now
demonstrated with two independent fine-tunes converging on the same result
rather than a single unverified before/after number.