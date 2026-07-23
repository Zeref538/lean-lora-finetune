"""Build v4 training data: self-distillation on the base model's own
verified-correct reasoning (distill_raw.jsonl), trimmed of boilerplate
intro/closing prose while preserving the actual derivation.

Unlike v1/v2 (compress GSM8K's already-terse gold labels to a fixed
near-minimal template), this keeps length proportional to problem
difficulty: only cut the parts that are the same regardless of the
problem (intro paragraph, closing restatement), never the math itself.

Usage:
    python data/build_v4.py
"""
import json
import re

from data.compress import extract_final_number

INTRO_SIGNALS = ("to determine", "to find", "to calculate", "let's break down", "we need to")


def trim(generation, gold):
    text = generation.strip()

    # drop a boilerplate intro paragraph: first chunk before the first blank
    # line, only if it reads like scene-setting (no actual math in it yet)
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        first, rest = parts
        first_lower = first.strip().lower()
        looks_like_intro = any(s in first_lower for s in INTRO_SIGNALS)
        has_math = "\\(" in first or "\\[" in first or re.search(r"\d+\s*[=+\-*/]\s*\d+", first)
        if looks_like_intro and not has_math:
            text = rest.strip()

    # drop a plan-only numbered list (e.g. "1. Calculate X. 2. Determine Y.")
    # when it's just an outline with no math, and something else follows it —
    # the plan gets re-executed below, so keeping both is pure duplication
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        first, rest = parts
        lines = [l.strip() for l in first.strip().split("\n") if l.strip()]
        is_numbered_list = len(lines) >= 2 and all(re.match(r"^\d+[.)]", l) for l in lines)
        has_math = "\\(" in first or "\\[" in first or re.search(r"\d+\s*[=+\-*/]\s*\d+", first)
        if is_numbered_list and not has_math:
            text = rest.strip()

    # replace the boilerplate closing ("Thus/Therefore ... \boxed{N}.") with
    # a plain "#### N" marker — same final-answer content, no restatement
    boxed = list(re.finditer(r"\\boxed\{([^}]*)\}", text))
    if boxed:
        last = boxed[-1]
        # find the start of the sentence/paragraph containing the boxed answer
        prior = text[:last.start()]
        cut_at = max(prior.rfind("\n\n"), prior.rfind(". ") + 1)
        cut_at = max(cut_at, 0)
        body = text[:cut_at].strip()
        text = body + ("\n" if body else "") + f"#### {gold}"

    return text


def main():
    kept, rejected = 0, 0
    raw_words, trimmed_words = [], []
    with open("data/distill_raw.jsonl", encoding="utf-8") as f_in, \
         open("data/pairs_v4.jsonl", "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            gen, gold = row["generation"].strip(), row["gold"]
            if extract_final_number(gen) != gold:
                continue  # self-distillation: only train on verified-correct output
            trimmed = trim(gen, gold)
            if extract_final_number(trimmed) != gold:
                rejected += 1
                continue
            f_out.write(json.dumps({"prompt": row["question"], "completion": trimmed}) + "\n")
            kept += 1
            raw_words.append(len(gen.split()))
            trimmed_words.append(len(trimmed.split()))

    avg_raw = sum(raw_words) / len(raw_words)
    avg_trim = sum(trimmed_words) / len(trimmed_words)
    print(f"kept={kept} rejected_by_trim={rejected}")
    print(f"avg words: raw={avg_raw:.1f} trimmed={avg_trim:.1f} ({(1 - avg_trim/avg_raw):.1%} shorter)")


if __name__ == "__main__":
    main()
