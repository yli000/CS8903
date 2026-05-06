# README

## how to run

### environment

Python 3.12, GPU host with an NVIDIA driver supporting CUDA 13.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu130
```


### data and model

```bash
hf download Qwen/Qwen3.5-9B --local-dir ./model/Qwen3.5-9B
hf download SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual --repo-type dataset --local-dir ./data
```

Place the trained LoRA adapter at `./models/<run-name>/`, and update `LORA_PATH` in `inference.py` to match. An empty string disables LoRA.

Expected layout:
```
./model/Qwen3.5-9B/              base model
./models/<run-name>/             LoRA adapter
./data/dataset_info.json         dataset JSON
./data/images/                   images
```

All paths are configured via globals at the top of `inference.py`:

| variable | meaning |
|---|---|
| `MODEL_PATH` | base model directory |
| `LORA_PATH` | LoRA adapter directory (`""` disables) |
| `MAX_LORA_RANK` | must match the rank used at training time |
| `DATA_JSON_PATH` | dataset JSON |
| `IMAGE_DIR` | image folder |
| `OUTPUT_JSON_PATH` | raw model output |
| `SUBMISSION_JSON_PATH` | competition submission output |
| `TEST_MODE` | `True` runs only `TEST_SAMPLE_COUNT` samples |

### run

```bash
bash inference.sh
```

Writes `./results/predictions.json` (raw output) and `./results/submission.json` (competition format). `INVALID` means the model exhausted its reasoning budget without emitting a final A-E choice; the submission writer falls back to `"A"` so the file always validates.

## results

### CLEF

Official results released after May 10 (date may be extended).

### Local

baseline: 0.7757 (first GRPO run shows no improvement)

## next steps

RL experiments for model's over-thinking problem (see report).
