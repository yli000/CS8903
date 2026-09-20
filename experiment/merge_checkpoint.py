# 把 verl 的 checkpoint 合并成 vLLM 能加载的完整模型。
#   A. verl.model_merger 把 FSDP 分片转成 HuggingFace base 模型加一个 lora_adapter/ 子目录
#   B. peft 加载 base + adapter，merge_and_unload 后保存完整权重
# 两步都幂等，目标目录里已有 config.json 就跳过。

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def merge_fsdp_shards(ckpt_dir: Path, base_dir: Path):
    if (base_dir / "config.json").exists():
        log(f"[A] skip, already merged at {base_dir}")
        return
    log(f"[A] verl.model_merger {ckpt_dir} -> {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(ckpt_dir),
        "--target_dir", str(base_dir),
    ]
    log("[A] " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"verl model_merger failed rc={rc}")
    log("[A] done")


def merge_lora_adapter(base_dir: Path, full_dir: Path):
    if (full_dir / "config.json").exists():
        log(f"[B] skip, already merged at {full_dir}")
        return

    lora_dir = base_dir / "lora_adapter"
    if not lora_dir.exists():
        # 没有 adapter，base 本身就是完整权重
        log(f"[B] no lora_adapter at {lora_dir}; base is already complete, symlinking")
        full_dir.parent.mkdir(parents=True, exist_ok=True)
        if not full_dir.exists():
            os.symlink(base_dir, full_dir)
        return

    log(f"[B] peft merge: base={base_dir} + adapter={lora_dir} -> {full_dir}")
    import gc
    import shutil

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    log("[B] loading base model in bf16 on CPU")
    base = AutoModelForImageTextToText.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    log("[B] attaching LoRA adapter")
    model = PeftModel.from_pretrained(base, str(lora_dir))
    log("[B] merge_and_unload")
    model = model.merge_and_unload()
    full_dir.mkdir(parents=True, exist_ok=True)
    log(f"[B] saving merged model to {full_dir}")
    model.save_pretrained(str(full_dir), safe_serialization=True)
    del model, base
    gc.collect()

    # tokenizer 和 chat template 一起带过去，vLLM 端要用
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "processor_config.json", "chat_template.jinja"):
        src = base_dir / name
        dst = full_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    log("[B] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, type=Path,
                    help="verl checkpoint 目录，形如 .../global_step_138/actor")
    ap.add_argument("--base_dir", required=True, type=Path,
                    help="第一步的输出：HF 格式 base 模型 + lora_adapter/")
    ap.add_argument("--full_dir", required=True, type=Path,
                    help="第二步的输出：合并后的完整模型，评测时指向这里")
    args = ap.parse_args()

    merge_fsdp_shards(args.ckpt_dir, args.base_dir)
    merge_lora_adapter(args.base_dir, args.full_dir)
    log(f"ready for inference: {args.full_dir}")


if __name__ == "__main__":
    main()
