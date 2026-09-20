# 输出 verl schema 的 parquet：
#   train.parquet ← EXAMS-V (train + validation)，过滤 Chinese + 指定学科 + image_text + 有效 answer
#   val.parquet   ← ImageCLEF-MR2026-MCQ-Visual (test)，过滤 Chinese + 同一批学科

import argparse
import io
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image as PILImage

# 常量
SEED = 42
MAX_IMAGE_SIZE = 1024
JPEG_QUALITY = 90
DATA_SOURCE = "exams-v-zh"

PROMPT_TEXT = "<image>\n分析这道题的题目、所有选项和存在的视觉元素（如有），仅回答一个选项：A、B、C、D或E。"

# 23 种 answer_key → A/B/C/D/E
ANSWER_MAP = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
    "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E",   # Cyrillic upper
    "а": "A", "б": "B", "в": "C", "г": "D", "д": "E",   # Cyrillic lower
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
}

# Subject 跨语言归一化
SUBJECT_CANON = {
    # 物理
    "物理": "Physics", "物理学": "Physics",
    "Physics": "Physics", "Physics and Astronomy": "Physics",
    "Física": "Physics", "Fizika": "Physics", "Fisica": "Physics",
    "Физика": "Physics",
    # 化学
    "化学": "Chemistry",
    "Chemistry": "Chemistry", "Chemistry and Environmental Protection": "Chemistry",
    "Химия": "Chemistry", "Химия и опазване на околната среда": "Chemistry",
    "Kemija": "Chemistry", "Chimica": "Chemistry",
    # 生物
    "生物": "Biology", "生物学": "Biology",
    "Biology": "Biology", "Biology and Health Education": "Biology",
    "Биология": "Biology", "Biologija": "Biology", "Biologia": "Biology",
    # 数学
    "数学": "Mathematics", "Mathematics": "Mathematics",
    "Математика": "Mathematics", "Matematika": "Mathematics", "Matematica": "Mathematics",
    # 地理
    "地理": "Geography",
    "Geography": "Geography", "Geography and Economics": "Geography",
    "География": "Geography", "Geografija": "Geography", "Geografia": "Geography",
    # 历史
    "历史": "History", "History": "History", "History and Civilizations": "History",
    "История": "History", "Povijest": "History", "Storia": "History", "Историја": "History",
    # 信息技术
    "计算机科学": "Informatics", "信息学": "Informatics",
    "Informatics": "Informatics", "Information Technology": "Informatics",
    "Информатика": "Informatics",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canon_subject(s):
    return SUBJECT_CANON.get(str(s).strip(), str(s).strip())


def map_answer(v):
    return ANSWER_MAP.get(str(v).strip(), "")


def encode_image(img):
    """PIL Image → JPEG bytes (压缩，对齐 grpo0)"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def find_answer_field(example_keys):
    """官方测试集 label 字段名可能是 answer / answer_key / label。"""
    for cand in ("answer_key", "answer", "label", "correct", "correct_answer"):
        if cand in example_keys:
            return cand
    return None


def build_row(image_bytes, gt, sample_id, subject, language, grade="", split=""):
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": PROMPT_TEXT}],
        "images": [{"bytes": image_bytes, "path": None}],
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "sample_id": str(sample_id),
            "subject": subject,
            "language": language,
            "grade": grade,
            "split": split,
        },
    }


def filter_and_build_train(target_subjects, all_zh):
    """从 EXAMS-V train + validation 抽中文样本。"""
    log("加载 EXAMS-V (train + validation)")
    ds = load_dataset("MBZUAI/EXAMS-V")
    log(f"  train={len(ds['train'])}  validation={len(ds['validation'])}")

    train_rows = []
    drop_stats = Counter()

    for split_name in ["train", "validation"]:
        split_ds = ds[split_name]
        cols = split_ds.column_names

        # 一次性列扫，确定要保留的 row index
        types = split_ds["type"]
        answers = split_ds["answer_key"]
        subjects = split_ds["subject"]
        langs = split_ds["language"]
        sids = split_ds["sample_id"] if "sample_id" in cols else [None] * len(split_ds)
        grades = split_ds["grade"] if "grade" in cols else [""] * len(split_ds)

        for i in range(len(split_ds)):
            if types[i] != "image_text":
                drop_stats["not_image_text"] += 1
                continue
            if langs[i] != "Chinese":
                drop_stats["not_chinese"] += 1
                continue
            subj_canon = canon_subject(subjects[i])
            if not all_zh and target_subjects and subj_canon not in target_subjects:
                drop_stats["wrong_subject"] += 1
                continue
            gt = map_answer(answers[i])
            if not gt:
                drop_stats["bad_answer"] += 1
                continue

            example = split_ds[i]
            img_bytes = encode_image(example["image"])
            sample_id = sids[i] or f"{split_name}-{i}"
            train_rows.append(build_row(
                img_bytes, gt, sample_id, subj_canon, langs[i],
                grade=grades[i] or "", split=split_name,
            ))

    log(f"  train_rows={len(train_rows)}  dropped={dict(drop_stats)}")
    return train_rows


def filter_and_build_val(target_subjects, all_zh):
    """从官方测试集 SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual 抽中文样本。

    用 hf_hub_download 直接拉 parquet 文件（不用 load_dataset，因为新版 datasets
    对 gated dataset 检查太严，即使权限有也报错）。
    """
    log("加载 SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual (via hf_hub_download)")
    from huggingface_hub import hf_hub_download
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    parquet_path = hf_hub_download(
        repo_id="SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    log(f"  下载完成: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    log(f"  test={len(df)} cols={list(df.columns)}")

    ans_field = find_answer_field(df.columns)
    if ans_field is None:
        log("  ❌ 找不到 answer 字段，官方测试集可能还没添加 label")
        log(f"  现有列: {list(df.columns)}")
        sys.exit(1)
    log(f"  使用 answer 字段: {ans_field}")

    # 逐行转成 dict 处理
    val_rows = []
    drop_stats = Counter()
    for i, row in df.iterrows():
        example = dict(row)
        # image 列可能是 bytes 或 PIL，统一处理为 PIL
        img = example.get("image")
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO
            example["image"] = PILImage.open(BytesIO(img["bytes"]))
        elif isinstance(img, bytes):
            from io import BytesIO
            example["image"] = PILImage.open(BytesIO(img))
        # 其余情况已经是 PIL 对象或路径
        if example.get("language") != "Chinese":
            drop_stats["not_chinese"] += 1
            continue
        subj_canon = canon_subject(example.get("subject", ""))
        if not all_zh and target_subjects and subj_canon not in target_subjects:
            drop_stats["wrong_subject"] += 1
            continue
        gt = map_answer(example.get(ans_field))
        if not gt:
            drop_stats["bad_answer"] += 1
            continue

        img_bytes = encode_image(example["image"])
        sample_id = example.get("sample_id") or example.get("id") or f"test-{i}"
        val_rows.append(build_row(
            img_bytes, gt, sample_id, subj_canon,
            example.get("language"),
            grade=str(example.get("grade", "")),
            split="official_test",
        ))

    log(f"  val_rows={len(val_rows)}  dropped={dict(drop_stats)}")
    return val_rows


def write_parquet(rows, path):
    if not rows:
        log(f"  ⚠️  空数据集，跳过 {path}")
        return
    # 保持 extra_info 为 dict（verl 要求 nested struct，不是 JSON string）
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    log(f"  写出 {path}  rows={len(rows)}  size={os.path.getsize(path)/1e6:.1f}MB")


def report(rows, label):
    if not rows:
        return
    subj = Counter(r["extra_info"]["subject"] for r in rows)
    lang = Counter(r["extra_info"]["language"] for r in rows)
    ans = Counter(r["reward_model"]["ground_truth"] for r in rows)
    log(f"  [{label}] subject={dict(subj)}  language={dict(lang)}  answer={dict(sorted(ans.items()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="", help="单学科 (canonical, eg Physics)")
    parser.add_argument("--subjects", type=str, default="", help="多学科逗号分隔")
    parser.add_argument("--all_zh", action="store_true", help="全部中文")
    parser.add_argument("--train_samples", type=int, default=-1)
    parser.add_argument("--val_samples", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="parquet 输出目录；必须显式指定，避免写到上一次实验的目录里")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.subjects:
        target_subjects = {s.strip() for s in args.subjects.split(",")}
    elif args.subject:
        target_subjects = {args.subject}
    else:
        target_subjects = None

    if args.all_zh:
        log("模式：全部中文")
    elif target_subjects:
        log(f"模式：限定学科 {target_subjects}")
    else:
        log("模式：未指定学科限制（建议加 --subject Physics）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练集
    train_rows = filter_and_build_train(target_subjects, args.all_zh)
    random.shuffle(train_rows)
    if args.train_samples > 0:
        train_rows = train_rows[: args.train_samples]
    report(train_rows, "train")
    write_parquet(train_rows, out_dir / "train.parquet")

    # 验证集
    val_rows = filter_and_build_val(target_subjects, args.all_zh)
    if args.val_samples > 0:
        val_rows = val_rows[: args.val_samples]
    report(val_rows, "val")
    write_parquet(val_rows, out_dir / "val.parquet")

    log(f"DONE → {out_dir}")


if __name__ == "__main__":
    main()
