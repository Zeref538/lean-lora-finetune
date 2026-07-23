"""
Compression-distillation data pipeline (TOKEN_EFFICIENCY_PLAN.md Phase 1).

For each (prompt, verbose_output):
  1. teacher rewrites verbose_output to its minimal complete form
  2. keep the pair only if the compressed output still passes the task's
     mechanical correctness check
  3. log the rejection rate

Usage:
    python compress.py --in raw.jsonl --out pairs.jsonl --task gsm8k
"""
import argparse
import json
import re

COMPRESS_PROMPT = """Rewrite the answer below to be as short as possible while
staying fully correct and complete. Remove restating the question, hedging,
filler ("Certainly!", "Let's think step by step"), and any reasoning steps
that aren't needed to justify the final answer. Keep the final answer intact.

Question: {question}
Answer: {answer}

Rewritten (short) answer:"""


def teacher_compress(client, question: str, answer: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": COMPRESS_PROMPT.format(question=question, answer=answer)}],
    )
    return msg.content[0].text.strip()


# --- mechanical correctness checks, one per task family ---
# Add new families here; each returns True if `output` is still correct
# against `gold`.
CHECKS = {
    "gsm8k": lambda output, gold: extract_final_number(output) == extract_final_number(gold),
    "json_exact": lambda output, gold: json_loose_equal(output, gold),
}


def extract_final_number(text: str):
    nums = re.findall(r"-?\d[\d,]*\.\d+|-?\d[\d,]*", text.replace(",", ""))
    return nums[-1] if nums else None


def json_loose_equal(output: str, gold: str) -> bool:
    try:
        return json.loads(output) == json.loads(gold)
    except (json.JSONDecodeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSONL of {question, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=CHECKS, required=True)
    args = ap.parse_args()

    check = CHECKS[args.task]
    import anthropic
    client = anthropic.Anthropic()

    kept, rejected = 0, 0
    with open(args.inp, encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            question, verbose = row["question"], row["answer"]
            short = teacher_compress(client, question, verbose)
            if check(short, verbose):
                f_out.write(json.dumps({"prompt": question, "completion": short}) + "\n")
                kept += 1
            else:
                rejected += 1

    total = kept + rejected
    print(f"kept={kept} rejected={rejected} rejection_rate={rejected/total:.1%}" if total else "no input rows")


if __name__ == "__main__":
    main()
