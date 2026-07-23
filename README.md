# Lean

Fine-tune a free open-weights model (Qwen2.5-1.5B-Instruct) via LoRA to say
the same thing in fewer tokens. **Result: it works, but not for free** — see
[CASE_STUDY.md](CASE_STUDY.md) for the full write-up, including a real
accuracy/brevity trade-off (69%→~50% accuracy for ~70% fewer tokens) and a
two-adapter ablation that isolates the actual cause. Also see
[../TOKEN_EFFICIENCY_PLAN.md](../TOKEN_EFFICIENCY_PLAN.md) for the original plan.

## Pipeline (as actually run)

1. **Data prep** — pulled the full GSM8K train split (7,473 rows) via
   `datasets.load_dataset`, then mechanically compressed it two ways to test
   a formatting hypothesis (`data/pairs.jsonl` = v1, lines joined with `; `;
   `data/pairs_v2.jsonl` = v2, original multi-line structure kept). Both
   verified via `data/compress.py`'s `CHECKS["gsm8k"]` gate (final-number
   match) — 0% rejection on both. `data/compress.py`'s teacher-rewrite path
   (`teacher_compress`, needs `ANTHROPIC_API_KEY`) was **not used** for this
   run — no API key/credits were used, only free mechanical compression.
2. **Train on Kaggle** (free T4 GPU) — `train_kaggle.ipynb` (v1 data) and
   `train_kaggle_v2.ipynb` (v2 data), both LoRA r=16 over 2 epochs. Download
   the resulting `adapter/` folder into `runs/lean-v1/` or `runs/lean-v2/`.
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
   python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --eval data/eval.jsonl --task gsm8k
   ```

## Test

`python data/test_compress.py` checks the correctness gates
(`extract_final_number`, `CHECKS`) — the only non-trivial logic here.
