# 给 run_inference.py 的输出打分。抽取规则与训练 reward 一致：
# 必须有闭合的 </think>，再在其后最后 5 个非空行里找 ANSWER:X / 答案：X / 单独一行的 A-E。
# 抽不出来记为 INVALID，计入分母算错。

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from normalize_subject import normalize_subject

VALID = {"A", "B", "C", "D", "E"}

# answer_key 的 23 种写法：拉丁字母、西里尔字母、数字，大小写都有
ANSWER_MAP = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
    "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E",
    "а": "A", "б": "B", "в": "C", "г": "D", "д": "E",
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
}

_ANSWER_RE = re.compile(r"(?:FINAL ANSWER|ANSWER|答案)\s*[:：]?\s*([A-E])\b", re.I)
_BARE_LETTER_RE = re.compile(r"^\s*([A-E])\s*$", re.M)


def map_answer_key(value):
    return ANSWER_MAP.get(str(value).strip(), "")


def extract_answer(text):
    if not text or "</think>" not in text:
        return "INVALID"
    _, tail = text.rsplit("</think>", 1)
    tail = tail.strip().upper()
    if not tail:
        return "INVALID"
    tail_lines = [line.strip() for line in tail.splitlines() if line.strip()]
    tail_text = "\n".join(tail_lines[-5:])
    m = _ANSWER_RE.search(tail_text)
    if m:
        return m.group(1).upper()
    m = _BARE_LETTER_RE.search(tail_text)
    if m:
        return m.group(1).upper()
    return "INVALID"


def make_bucket():
    return {"n": 0, "correct": 0, "invalid": 0}


def finalize(bucket):
    out = dict(bucket)
    out["accuracy"] = round(out["correct"] / out["n"], 4) if out["n"] else 0.0
    out["invalid_rate"] = round(out["invalid"] / out["n"], 4) if out["n"] else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path,
                    help="run_inference.py 产出的 JSONL")
    ap.add_argument("--out", type=Path, default=None,
                    help="把结果写成 JSON；省略则只打印")
    args = ap.parse_args()

    overall = make_bucket()
    by_language = defaultdict(make_bucket)
    by_subject = defaultdict(make_bucket)
    lengths = []

    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            completion = row.get("completion", "")
            predicted = extract_answer(completion)
            gt = map_answer_key(row.get("answer_key", ""))
            invalid = predicted not in VALID
            correct = (not invalid) and predicted == gt and gt in VALID
            lengths.append(len(completion))

            language = row.get("language", "") or "UNKNOWN"
            subject = normalize_subject(row.get("subject", "") or "UNKNOWN")
            for bucket in (overall, by_language[language], by_subject[subject]):
                bucket["n"] += 1
                bucket["invalid"] += invalid
                bucket["correct"] += correct

    result = {
        "overall": {
            **finalize(overall),
            "median_completion_chars": int(statistics.median(lengths)),
            "mean_completion_chars": round(statistics.mean(lengths)),
        },
        "by_language": {k: finalize(v) for k, v in sorted(by_language.items())},
        "by_subject": {k: finalize(v) for k, v in sorted(by_subject.items())},
    }

    o = result["overall"]
    print(f"n={o['n']}  accuracy={o['accuracy']:.4f}  invalid={o['invalid']} ({o['invalid_rate']:.4f})  "
          f"median_len={o['median_completion_chars']}  mean_len={o['mean_completion_chars']}")
    print("\nby language")
    for k, v in result["by_language"].items():
        print(f"  {k:12s} n={v['n']:4d}  accuracy={v['accuracy']:.4f}  invalid={v['invalid_rate']:.4f}")
    print("\nby subject")
    for k, v in result["by_subject"].items():
        print(f"  {k:14s} n={v['n']:4d}  accuracy={v['accuracy']:.4f}  invalid={v['invalid_rate']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
