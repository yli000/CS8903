English | [中文](README.zh-CN.md)

# Baseline

Zero-shot Qwen3.5 on the official test set: inference, scoring, submission formatting.

The same two scripts also score a fine-tuned checkpoint. Merge it first with [`../experiment/merge_checkpoint.py`](../experiment/merge_checkpoint.py), then point `--model` at the merged directory.

## Environment

Python 3.12, GPU host with an NVIDIA driver supporting CUDA 13.

`requirements.txt` is a full freeze including CUDA wheels, so it needs the extra index URLs.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

The test set is gated:

```bash
export HF_TOKEN=...
```

## Data and model

`run_inference.py` and `evaluate.py` need nothing downloaded by hand. `run_inference.py` fetches the test
parquet from the Hugging Face dataset and loads the model by name, so `HF_TOKEN` is enough.

`submit.py` reads a local copy instead, one JSON entry per question plus one image file per question.
Both datasets ship the images as bytes inside the parquet, so `extract_images.py` unpacks them first:

```bash
hf download Qwen/Qwen3.5-9B --local-dir ./model/Qwen3.5-9B

# either dataset works; this is the EXAMS-V validation split
hf download MBZUAI/EXAMS-V --repo-type dataset --local-dir ./data

python extract_images.py \
  --parquet './data/validation-*.parquet' \
  --out_json ./data/validation_data.json \
  --image_dir ./data/images \
  --gold_json ./data/validation_gold.json

python submit.py
```

`extract_images.py` writes each row's image to `<image_dir>/<id>.png`, replaces the row's `image` field
with that filename, and adds an `id` field taken from whichever of `sample_id`, `question_id` or `id` the
parquet uses. `--gold_json` is optional and writes an `id` → `answer` file for local scoring.

The resulting layout:

| path | contents |
|---|---|
| `./model/Qwen3.5-9B/` | base model |
| `./models/<run-name>/` | LoRA adapter, when `LORA_PATH` is set |
| `./data/validation_data.json` | one entry per question, keyed by `id` |
| `./data/images/` | one image per question, named by the entry's `image` field |

## Run

2× H100-80GB. The 6144-token run takes about 30 minutes.

```bash
# baseline at the same budget GRPO was trained under
python run_inference.py \
  --model Qwen/Qwen3.5-9B \
  --out predictions/base_9b_6k.jsonl
python evaluate.py \
  --predictions predictions/base_9b_6k.jsonl \
  --out results/base_9b_6k.json
```

Budget control — same model and decoding, cap lifted to Qwen3.5-9B's native 32k context: to explore whether "the model cannot solve it" or "the model ran out of room to write the answer down".

```bash
python run_inference.py \
  --model Qwen/Qwen3.5-9B \
  --out predictions/base_9b_32k.jsonl \
  --max_new_tokens 30000 --max_model_len 32768 --max_num_seqs 12 --tp_size 2
```

A merged GRPO checkpoint:

```bash
python run_inference.py \
  --model "$CLEF_PROJ/exps/zh-physics/merged_step138_full" \
  --out predictions/no_length_step138.jsonl
```

Add `--limit 20` to check the path end to end before spending a full job.

At 32k context the default concurrency will not fit in the KV cache, so `--max_num_seqs` has to come down and `--tp_size 2` spreads the cache over two GPUs. Setting `--max_num_seqs` too low instead makes the run miss a typical walltime. 12 sequences on 2 GPUs worked.

## Files

| File | Purpose |
|---|---|
| `run_inference.py` | Downloads the test parquet, runs vLLM greedy over all 1117 questions, writes one JSONL line per question with the raw completion. No scoring |
| `evaluate.py` | Applies the extraction rule, reports accuracy, invalid rate and completion length, overall and by language and subject |
| `normalize_subject.py` | 32 subject spellings → 9 canonical subjects. Used by `evaluate.py` |
| `extract_images.py` | Unpacks the images held as bytes inside a parquet into files, plus the JSON `submit.py` reads |
| `submit.py` | Submission inference. Resumable, writes `submission.json` in the official format after every batch |
| `requirements.txt` | Environment freeze |

`run_inference.py` writes metadata and the raw completion only. Re-scoring with a different rule needs `evaluate.py` alone, not another inference run.

`submit.py` is configured by the globals at the top of the file.

| variable | meaning |
|---|---|
| `MODEL_PATH` | base model directory |
| `LORA_PATH` | LoRA adapter directory. Empty string disables LoRA |
| `MAX_LORA_RANK` | must match the rank used at training time (64), or vLLM refuses the adapter |
| `DATA_JSON_PATH` | dataset JSON |
| `IMAGE_DIR` | image folder |
| `OUTPUT_JSON_PATH` | raw model output |
| `SUBMISSION_JSON_PATH` | submission output |
| `TEST_MODE` | `True` runs only `TEST_SAMPLE_COUNT` samples |

`INVALID` means the model exhausted its reasoning budget without emitting a final A-E choice; the submission writer falls back to `"A"` so the file always validates.

## References

- Task organizers' repo — [src/baselines](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning/tree/main/2026/src/baselines) · [src/evaluation](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning/tree/main/2026/src/evaluation)
