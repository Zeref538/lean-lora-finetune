"""
Report accuracy AND output-token count together — a shorter wrong answer is
a failure, not a saving (TOKEN_EFFICIENCY_PLAN.md ground rule).

Usage:
    python eval.py --model runs/lean-v1/adapter --eval data/eval.jsonl --task gsm8k
"""
import argparse
import json

import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.compress import CHECKS, extract_final_number  # noqa: F401  (reuse correctness checks)

# CPU eval can't use the 4-bit bnb base the adapter was trained against;
# LoRA deltas apply the same on top of the full-precision base weights.
FULL_PRECISION_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a LoRA adapter dir, or a base model id/path")
    ap.add_argument("--eval", required=True, help="JSONL of {question, answer}")
    ap.add_argument("--task", choices=CHECKS, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if os.path.exists(os.path.join(args.model, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(FULL_PRECISION_BASE)
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    check = CHECKS[args.task]

    correct, total_tokens, n = 0, 0, 0
    for line in open(args.eval, encoding="utf-8"):
        row = json.loads(line)
        inputs = tok(row["question"], return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=512)
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)

        n += 1
        total_tokens += len(gen_tokens)
        if check(text, row["answer"]):
            correct += 1

    print(f"accuracy={correct/n:.1%}  mean_output_tokens={total_tokens/n:.1f}  n={n}")


if __name__ == "__main__":
    main()
