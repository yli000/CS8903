# 官方 Visual-MCQ test set 上的推理，输出原始 completion，不打分。

import argparse
import json
import os
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage

TEST_REPO = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
TEST_FILE = "data/test-00000-of-00001.parquet"

# 与 prep_data.py 训练时用的 prompt 一致
PROMPT_TEXT = "分析这道题的题目、所有选项和存在的视觉元素（如有），仅回答一个选项：A、B、C、D或E。"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_test_set(hf_token):
    from huggingface_hub import hf_hub_download

    log(f"downloading {TEST_REPO}:{TEST_FILE}")
    path = hf_hub_download(
        repo_id=TEST_REPO,
        filename=TEST_FILE,
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(path)
    log(f"loaded {len(df)} rows, cols={list(df.columns)}")
    return df


def to_pil(image_field):
    if isinstance(image_field, dict) and "bytes" in image_field:
        return PILImage.open(BytesIO(image_field["bytes"])).convert("RGB")
    if isinstance(image_field, (bytes, bytearray)):
        return PILImage.open(BytesIO(image_field)).convert("RGB")
    if isinstance(image_field, PILImage.Image):
        return image_field.convert("RGB")
    raise ValueError(f"unknown image type: {type(image_field)}")


def build_chat_prompt(processor):
    # 文本走 chat template，图片由 vLLM 的 multi_modal_data 传
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEXT}],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace 模型名或本地模型目录")
    ap.add_argument("--out", required=True, type=Path,
                    help="输出的 predictions JSONL 路径")
    ap.add_argument("--max_new_tokens", type=int, default=6144,
                    help="completion 上限；6144 与 GRPO 训练时的 max_response_length 一致")
    ap.add_argument("--max_model_len", type=int, default=8192,
                    help="vLLM 上下文上限（prompt + completion）")
    ap.add_argument("--max_num_seqs", type=int, default=64,
                    help="vLLM 并发序列数；长上下文要调小，否则 KV cache 装不下")
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 条，用来先验证流程通不通")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN is required: the test set is a gated dataset")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    df = load_test_set(hf_token)
    if args.limit > 0:
        df = df.head(args.limit)

    log(f"loading processor from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    log(f"loading vLLM (tp={args.tp_size}, max_model_len={args.max_model_len}, "
        f"max_num_seqs={args.max_num_seqs}, gpu_mem={args.gpu_mem_util})")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    prompt = build_chat_prompt(processor)
    requests = []
    metas = []
    for i, row in df.iterrows():
        requests.append({"prompt": prompt, "multi_modal_data": {"image": to_pil(row["image"])}})
        metas.append({
            "question_id": str(row.get("question_id", f"row-{i}")),
            "language": row.get("language", ""),
            "subject": row.get("subject", ""),
            "answer_key": row.get("answer_key", ""),
        })

    log(f"generating {len(requests)} completions")
    t0 = time.time()
    outputs = llm.generate(requests, sampling)
    log(f"generate done in {time.time() - t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for meta, output in zip(metas, outputs):
            rec = {**meta, "completion": output.outputs[0].text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"wrote {len(metas)} predictions to {args.out}")


if __name__ == "__main__":
    main()
