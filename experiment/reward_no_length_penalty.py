# 消融用的 reward：score = correctness + 0.05 * [有 </think>]
# 与 reward_length_penalty.py 的唯一区别是去掉了 length penalty 项。

import json
import os
import re
import threading
import time

_LOG_PATH = os.environ.get(
    "ROLLOUT_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_log_no_length.jsonl"),
)
_LOG_LOCK = threading.Lock()

FORMAT_BONUS = 0.05           # 闭合 </think> 的奖励
_ANSWER_RE = re.compile(r"(?:FINAL ANSWER|ANSWER|答案)\s*[:：]?\s*([A-E])\b", re.I)
_BARE_LETTER_RE = re.compile(r"^\s*([A-E])\s*$", re.M)


def extract_answer(text: str) -> str:
    if not text:
        return "INVALID"
    if "</think>" not in text:
        return "INVALID"

    _, answer_text = text.rsplit("</think>", 1)
    answer_text = answer_text.strip().upper()
    if not answer_text:
        return "INVALID"

    tail_lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    tail_text = "\n".join(tail_lines[-5:])

    m = _ANSWER_RE.search(tail_text)
    if m:
        return m.group(1).upper()
    m = _BARE_LETTER_RE.search(tail_text)
    if m:
        return m.group(1).upper()
    return "INVALID"


def _log(solution_str, ground_truth, predicted, score, response_len):
    record = {
        "t": time.time(),
        "gt": ground_truth.strip().upper(),
        "predicted": predicted,
        "score": score,
        "response_len": response_len,
        "response": solution_str,
    }
    with _LOG_LOCK:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    predicted = extract_answer(solution_str)
    gt = ground_truth.strip().upper()
    response_len = len(solution_str)

    is_correct = 1.0 if predicted == gt else 0.0

    has_closing = 1.0 if "</think>" in solution_str else 0.0
    format_bonus = FORMAT_BONUS * has_closing

    score = is_correct + format_bonus

    _log(solution_str, gt, predicted, score, response_len)

    head = solution_str[:80].replace("\n", "\\n")
    tail = solution_str[-80:].replace("\n", "\\n")
    print(
        f"[REWARD-NOLEN] gt={gt} pred={predicted} score={score:.3f} "
        f"correct={is_correct:.0f} fmt={format_bonus:.3f} "
        f"len={response_len} head={head!r} tail={tail!r}",
        flush=True,
    )

    return {
        "score": score,
        "predicted": predicted,
        "invalid": 1.0 if predicted == "INVALID" else 0.0,
    }
