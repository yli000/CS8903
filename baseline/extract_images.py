# 把 parquet 里的图片抽成文件，配一份 JSON，供 submit.py 读取。
# 逐批读写，避免把整个 parquet 载入内存。

import argparse
import gc
import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

BATCH_SIZE = 64

# 不同数据集用的主键名不一样
ID_FIELDS = ("sample_id", "question_id", "id")
ANSWER_FIELDS = ("answer_key", "answer")


def pick_field(item, candidates):
    for name in candidates:
        if item.get(name) not in (None, ""):
            return name
    return None


def write_start(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")


def append_item(path, item, is_first):
    with open(path, "a", encoding="utf-8") as f:
        if not is_first:
            f.write(",\n")
        json.dump(item, f, ensure_ascii=False)


def write_end(path):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n]\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="parquet 的 glob，例如 './data/validation-*.parquet'")
    ap.add_argument("--out_json", required=True, type=Path)
    ap.add_argument("--image_dir", required=True, type=Path)
    ap.add_argument("--gold_json", type=Path, default=None,
                    help="另外写一份 id → answer 的对照，用来本地打分")
    args = ap.parse_args()

    files = sorted(glob.glob(args.parquet))
    if not files:
        raise SystemExit(f"no parquet matched: {args.parquet}")

    args.image_dir.mkdir(parents=True, exist_ok=True)
    write_start(args.out_json)
    if args.gold_json:
        write_start(args.gold_json)

    first = True
    total = 0
    for path in files:
        print(f"processing {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
            for item in tqdm(batch.to_pylist(), desc=os.path.basename(path), leave=False):
                image_info = item.get("image") or {}
                image_bytes = image_info.get("bytes")
                if not image_bytes:
                    continue

                id_field = pick_field(item, ID_FIELDS)
                if id_field is None:
                    raise SystemExit(f"no id field among {ID_FIELDS} in {list(item)}")
                item_id = item[id_field]

                image_name = f"{item_id}.png"
                with open(args.image_dir / image_name, "wb") as f:
                    f.write(image_bytes)

                # submit.py 按 id 索引，按 image 字段找图片文件
                if image_info.get("path"):
                    item["image_path"] = image_info["path"]
                item["image"] = image_name
                item["id"] = item_id

                append_item(args.out_json, item, first)
                if args.gold_json:
                    answer_field = pick_field(item, ANSWER_FIELDS)
                    gold = {"id": item_id,
                            "answer": str(item[answer_field]).strip().upper() if answer_field else ""}
                    append_item(args.gold_json, gold, first)
                first = False
                total += 1
            gc.collect()

    write_end(args.out_json)
    if args.gold_json:
        write_end(args.gold_json)
    print(f"{total} records -> {args.out_json}")
    print(f"images -> {args.image_dir}")


if __name__ == "__main__":
    main()
