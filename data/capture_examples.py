"""One-off: capture real generations from base + adapters on a few held-out
questions, for the portfolio demo page (pre-recorded, no live model calls
needed at page-view time).

Usage:
    python data/capture_examples.py
"""
import json
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FULL_PRECISION_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
N_EXAMPLES = 4
MODELS = {
    "base": None,
    "v1": "runs/lean-v1/adapter",
    "v2": "runs/lean-v2/adapter",
}


def load(adapter_path):
    tok = AutoTokenizer.from_pretrained(adapter_path or FULL_PRECISION_BASE)
    base = AutoModelForCausalLM.from_pretrained(FULL_PRECISION_BASE)
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    return tok, model


def main():
    questions = [json.loads(l) for l in open("data/eval.jsonl", encoding="utf-8")][:N_EXAMPLES]

    results = {name: [] for name in MODELS}
    for name, adapter_path in MODELS.items():
        tok, model = load(adapter_path)
        for row in questions:
            inputs = tok(row["question"], return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=512)
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = tok.decode(gen_tokens, skip_special_tokens=True)
            results[name].append({"question": row["question"], "answer": text, "tokens": len(gen_tokens)})
        del model
        print(f"done: {name}")

    with open("data/demo_examples.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote data/demo_examples.json")


if __name__ == "__main__":
    main()
