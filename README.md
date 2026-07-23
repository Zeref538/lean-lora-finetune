# Lean

**[Live demo →](https://zeref538.github.io/lean-lora-finetune/)**

Fine-tune a free open-weights model (Qwen2.5-1.5B-Instruct) via LoRA to say
the same thing in fewer tokens, without giving up accuracy. Four fine-tunes,
two falsified hypotheses, and one recipe that actually held accuracy near
the base model's — see [CASE_STUDY.md](CASE_STUDY.md) for the full write-up.
Also see [../TOKEN_EFFICIENCY_PLAN.md](../TOKEN_EFFICIENCY_PLAN.md) for the
original plan.

## Results

| | Base | v1 (2ep, gold labels, collapsed) | v2 (2ep, gold labels, structured) | v3 (1ep, gold labels) | **v4 (2ep, self-distilled)** |
|---|---|---|---|---|---|
| Accuracy (n=100, held-out GSM8K test) | 69.0% | 51.0% | 50.0% | 45.0% | **64.0%** |
| Mean output tokens | 299.4 | 93.8 | 89.7 | 86.4 | **248.4** |

v1-v3 all trained on GSM8K's own gold-label style — uniformly ~52 words
regardless of problem difficulty — and all paid an 18-24 point accuracy
cost for a ~70% token reduction. Diagnosing why led to v4: **self-distill
on the base model's own verified-correct reasoning** (kept only where it
already got the right answer, 5,780 of 7,473 examples), trimming only
genuine boilerplate (intro/closing prose) while leaving the actual
derivation untouched — so length still scales with problem difficulty.
Result: accuracy held to within 5 points of base, at a real but more modest
~17% token reduction. Two hypotheses about v1-v3's cost were tested and
ruled out by evidence along the way:

- **Not caused by the compressor deleting reasoning** — training labels
  were barely shortened from GSM8K's originals (~0% word reduction).
- **Not caused by reasoning-line-joining format** — v1 (semicolon-joined)
  and v2 (original multi-line structure) landed statistically identical.
- **Not a smooth less-training-preserves-more-accuracy dial either** — v3
  (1 epoch) did *worse* on both axes than v1/v2 (2 epochs), the opposite of
  that hypothesis.

Full reasoning, the self-distillation recipe, and why it works is in
[CASE_STUDY.md](CASE_STUDY.md).

## Pipeline (as actually run)

1. **Data prep** — pulled the full GSM8K train split (7,473 rows) via
   `datasets.load_dataset`, then mechanically compressed it two ways to test
   a formatting hypothesis (`data/pairs.jsonl` = v1, lines joined with `; `;
   `data/pairs_v2.jsonl` = v2, original multi-line structure kept). Both
   verified via `data/compress.py`'s `CHECKS["gsm8k"]` gate (final-number
   match) — 0% rejection on both. `data/compress.py`'s teacher-rewrite path
   (`teacher_compress`, needs `ANTHROPIC_API_KEY`) was **not used** for this
   run — no API key/credits were used, only free mechanical compression.
2. **Self-distillation for v4** (free T4 GPU) — `distill_generate.ipynb`
   batch-generates the base model's own reasoning on all 7,473 training
   questions → `data/distill_raw.jsonl`. `data/build_v4.py` keeps only
   verified-correct generations (77.3%) and trims boilerplate intro/closing
   prose only, leaving the derivation untouched → `data/pairs_v4.jsonl`
   (26.5% shorter, 0% rejected by the trim).
3. **Train on Kaggle** (free T4 GPU):
   - `train_kaggle.ipynb` → v1 data, 2 epochs → `runs/lean-v1/adapter/`
   - `train_kaggle_v2.ipynb` → v2 data, 2 epochs → `runs/lean-v2/adapter/`
   - `train_kaggle_v3.ipynb` → v2 data, 1 epoch → `runs/lean-v3/adapter/`
   - `train_kaggle_v4.ipynb` → v4 self-distilled data, 2 epochs → `runs/lean-v4/adapter/`

   All LoRA r=16, Unsloth-accelerated, on Qwen2.5-1.5B-Instruct-bnb-4bit.
   `train.py` + `configs/qwen_lora.yaml` is the same run expressed as a
   local/reproducible script, if you have a GPU handy instead.
4. `eval.py` — reports accuracy **and** mean output tokens together on a
   held-out set (GSM8K test split, disjoint from training); a shorter wrong
   answer is a failure, not a saving. Loads a LoRA adapter dir by applying
   it over the full-precision base model (works on CPU, unlike the 4-bit
   quantized base the adapter trained against). `eval_kaggle.ipynb` runs the
   same measurement batched on GPU — much faster than the sequential CPU
   loop below for multiple models at once.
   ```
   python eval.py --model runs/lean-v1/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model runs/lean-v2/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model runs/lean-v3/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model runs/lean-v4/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --eval data/eval.jsonl --task gsm8k
   ```
5. `data/capture_examples.py` — captures real, pre-recorded generations from
   each model on a few held-out questions, saved to `data/demo_examples.json`
   for a portfolio demo (no live model calls needed to show the comparison).

## Test

`python data/test_compress.py` checks the correctness gates
(`extract_final_number`, `CHECKS`) — the only non-trivial logic here.
