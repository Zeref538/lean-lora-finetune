# Lean

Fine-tune a free open-weights model (Qwen2.5-1.5B-Instruct) via LoRA to say
the same thing in fewer tokens. **Result: it works, but not for free** — see
[CASE_STUDY.md](CASE_STUDY.md) for the full write-up: a real accuracy/brevity
trade-off, and two hypotheses about its cause tested and falsified across
three independent fine-tunes. Also see
[../TOKEN_EFFICIENCY_PLAN.md](../TOKEN_EFFICIENCY_PLAN.md) for the original plan.

## Results

| | Base model | v1 (2 epochs, line-collapsed) | v2 (2 epochs, structure-preserved) | v3 (1 epoch, structure-preserved) |
|---|---|---|---|---|
| Accuracy (n=100, held-out GSM8K test) | 69.0% | 51.0% | 50.0% | 45.0% |
| Mean output tokens | 299.4 | 93.8 | 89.7 | 86.4 |

All three fine-tunes cut output tokens ~70% relative to the untouched base
model, at a real, confirmed accuracy cost. Two hypotheses about *why* were
tested and ruled out by evidence:

- **Not caused by the compressor deleting reasoning** — training labels
  were barely shortened from GSM8K's originals (~0% word reduction).
- **Not caused by reasoning-line-joining format** — v1 (semicolon-joined)
  and v2 (original multi-line structure) landed statistically identical.
- **Not a smooth less-training-preserves-more-accuracy dial either** — v3
  (1 epoch) did *worse* on both axes than v1/v2 (2 epochs), the opposite of
  that hypothesis.

Full reasoning and the best-supported explanation (a style/reasoning-
recovery timescale split during SFT) is in [CASE_STUDY.md](CASE_STUDY.md).

## Pipeline (as actually run)

1. **Data prep** — pulled the full GSM8K train split (7,473 rows) via
   `datasets.load_dataset`, then mechanically compressed it two ways to test
   a formatting hypothesis (`data/pairs.jsonl` = v1, lines joined with `; `;
   `data/pairs_v2.jsonl` = v2, original multi-line structure kept). Both
   verified via `data/compress.py`'s `CHECKS["gsm8k"]` gate (final-number
   match) — 0% rejection on both. `data/compress.py`'s teacher-rewrite path
   (`teacher_compress`, needs `ANTHROPIC_API_KEY`) was **not used** for this
   run — no API key/credits were used, only free mechanical compression.
2. **Train on Kaggle** (free T4 GPU):
   - `train_kaggle.ipynb` → v1 data, 2 epochs → `runs/lean-v1/adapter/`
   - `train_kaggle_v2.ipynb` → v2 data, 2 epochs → `runs/lean-v2/adapter/`
   - `train_kaggle_v3.ipynb` → v2 data, 1 epoch → `runs/lean-v3/adapter/`

   All LoRA r=16, Unsloth-accelerated, on Qwen2.5-1.5B-Instruct-bnb-4bit.
   `train.py` + `configs/qwen_lora.yaml` is the same run expressed as a
   local/reproducible script, if you have a GPU handy instead.
3. `eval.py` — reports accuracy **and** mean output tokens together on a
   held-out set (GSM8K test split, disjoint from training); a shorter wrong
   answer is a failure, not a saving. Loads a LoRA adapter dir by applying
   it over the full-precision base model (works on CPU, unlike the 4-bit
   quantized base the adapter trained against).
   ```
   python eval.py --model runs/lean-v1/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model runs/lean-v2/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model runs/lean-v3/adapter --eval data/eval.jsonl --task gsm8k
   python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --eval data/eval.jsonl --task gsm8k
   ```
4. `data/capture_examples.py` — captures real, pre-recorded generations from
   base/v1/v2 on a few held-out questions, saved to `data/demo_examples.json`
   for a portfolio demo (no live model calls needed to show the comparison).

## Test

`python data/test_compress.py` checks the correctness gates
(`extract_final_number`, `CHECKS`) — the only non-trivial logic here.
