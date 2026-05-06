import base64
import json
import multiprocessing
import os
import re

MODEL_PATH = "./model/Qwen3.5-9B"
LORA_PATH = "./models/<PLACEHOLDER>"
DATA_JSON_PATH = "./data/validation_data.json"
IMAGE_DIR = "./data/images"
OUTPUT_JSON_PATH = "./results/predictions.json"
SUBMISSION_JSON_PATH = "./results/submission.json"
BATCH_SIZE = 50
TEST_MODE = False
TEST_SAMPLE_COUNT = 5
MAX_LORA_RANK = 16
INVALID_FALLBACK = "A"
VALID_ANSWERS = {"A", "B", "C", "D", "E"}

PROMPT_TEXT = """分析这道题的题目、所有选项和存在的视觉元素（如有），仅回答一个选项：A、B、C、D或E。"""


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["id"]: item for item in data}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_results(path, results_dict):
    results_list = list(results_dict.values())
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_submission(results_dict, path):
    submission = [
        {
            "question_id": item["id"],
            "answer_key": item["prediction"] if item["prediction"] in VALID_ANSWERS else INVALID_FALLBACK,
        }
        for item in results_dict.values()
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)


def parse_output(output):
    full_text = output.outputs[0].text.strip()
    reasoning = ""
    answer_text = ""

    if "</think>" in full_text:
        reasoning_text, answer_text = full_text.rsplit("</think>", 1)
        reasoning = reasoning_text.replace("<think>", "").strip()
    elif "<think>" in full_text:
        reasoning = full_text.split("<think>", 1)[1].strip()
        return "INVALID", reasoning
    else:
        return "INVALID", ""

    answer_text = answer_text.strip().upper()
    tail_lines = [line.strip().upper() for line in answer_text.splitlines() if line.strip()]
    tail_text = "\n".join(tail_lines[-5:])

    exact_match = (
        re.search(r"(?:FINAL ANSWER|ANSWER|答案)\s*[:：]?\s*([A-E])\b", tail_text, re.I)
        or re.search(r"^\s*([A-E])\s*$", tail_text, re.M)
    )
    prediction = exact_match.group(1) if exact_match else "INVALID"
    return prediction, reasoning


def build_message(image_path):
    image_base64_url = image_to_base64(image_path)
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT_TEXT},
            {"type": "image_url", "image_url": {"url": image_base64_url}},
        ],
    }]


def normalize_omp_num_threads():
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if re.fullmatch(r"[1-9]\d*", omp_num_threads or ""):
        return
    os.environ["OMP_NUM_THREADS"] = "1"


def main():
    normalize_omp_num_threads()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    multiprocessing.set_start_method("spawn", force=True)

    all_data = load_data(DATA_JSON_PATH)
    if TEST_MODE:
        print(f"Test mode enabled: only running first {TEST_SAMPLE_COUNT} samples")
        all_data = all_data[:TEST_SAMPLE_COUNT]

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_JSON_PATH), exist_ok=True)
    existing = load_existing_results(OUTPUT_JSON_PATH)
    pending_list = [item for item in all_data if item["id"] not in existing]

    print(f"Total samples: {len(all_data)}")
    print(f"Already finished: {len(existing)}")
    print(f"Pending: {len(pending_list)}")

    if not pending_list:
        write_submission(existing, SUBMISSION_JSON_PATH)
        print(f"Submission file refreshed: {SUBMISSION_JSON_PATH}")
        return

    print("Loading model...")
    llm_kwargs = dict(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1},
    )
    if LORA_PATH:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = MAX_LORA_RANK
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    lora_request = LoRARequest("adapter", 1, LORA_PATH) if LORA_PATH else None

    total_batches = (len(pending_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(pending_list))
        batch = pending_list[start:end]
        batch_items = []
        batch_messages = []

        print(f"Batch {batch_idx + 1}/{total_batches} ({len(batch)} samples)")

        for item in batch:
            image_name = item.get("image") or f"{item['id']}.png"
            image_path = os.path.join(IMAGE_DIR, image_name)
            if not os.path.exists(image_path):
                print(f"Missing image, skipped: {image_path}")
                continue
            batch_items.append(item)
            batch_messages.append(build_message(image_path))

        if not batch_messages:
            continue

        outputs = llm.chat(
            messages=batch_messages,
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
        )

        for item, output in zip(batch_items, outputs):
            prediction, reasoning = parse_output(output)
            existing[item["id"]] = {
                "id": item["id"],
                "prediction": prediction,
                "reasoning": reasoning,
            }

        save_results(OUTPUT_JSON_PATH, existing)
        write_submission(existing, SUBMISSION_JSON_PATH)
        print(f"Saved {len(existing)} predictions")

    print(f"Predictions saved to: {OUTPUT_JSON_PATH}")
    print(f"Submission saved to: {SUBMISSION_JSON_PATH}")


if __name__ == "__main__":
    main()
