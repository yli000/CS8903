[English](README.md) | 中文

# Baseline

Qwen3.5 零样本跑官方测试集：推理、打分、提交格式。

微调出来的 checkpoint 也用这两个脚本打分。先用 [`../experiment/merge_checkpoint.py`](../experiment/merge_checkpoint.py) 合并，再把 `--model` 指向合并结果的目录。

## 环境

Python 3.12，GPU 机器的 NVIDIA 驱动要支持 CUDA 13。

`requirements.txt` 是完整冻结，包含 CUDA 的 wheel，所以要带上额外的 index URL。

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

测试集需要权限：

```bash
export HF_TOKEN=...
```

## 数据与模型

`run_inference.py` 和 `evaluate.py` 不需要手动下载任何东西。`run_inference.py` 会自己从 Hugging Face 拉测试集的 parquet，模型按名字加载，有 `HF_TOKEN` 就够。

`submit.py` 读的是本地副本，需要下面这个结构：

| 路径 | 内容 |
|---|---|
| `./model/Qwen3.5-9B/` | base 模型 —— `hf download Qwen/Qwen3.5-9B --local-dir ./model/Qwen3.5-9B` |
| `./models/<run-name>/` | LoRA adapter，设了 `LORA_PATH` 时需要 |
| `./data/validation_data.json` | 每题一条，用 `id` 作为键 |
| `./data/images/` | 每题一张图，文件名取该条的 `image` 字段，没有这个字段时用 `<id>.png` |

Hugging Face 上的数据集只发布一个 parquet 文件 `data/test-00000-of-00001.parquet`，得不到上面这个结构。`submit.py` 对应的是任务方自己发布的那份数据，与他们 `src/baselines/` 脚本接受的 JSON 加图片目录结构相同。

## 运行

2 张 H100-80GB。6144 token 大约 30 分钟。

```bash
# baseline，预算与 GRPO 训练时一致
python run_inference.py \
  --model Qwen/Qwen3.5-9B \
  --out predictions/base_9b_6k.jsonl
python evaluate.py \
  --predictions predictions/base_9b_6k.jsonl \
  --out results/base_9b_6k.json
```

预算对照 —— 同一个模型、同样的解码设置，上限放宽到 Qwen3.5-9B 原生的 32k 上下文：为了看看是「模型解不出来」还是「没有余量把答案写完」。

```bash
python run_inference.py \
  --model Qwen/Qwen3.5-9B \
  --out predictions/base_9b_32k.jsonl \
  --max_new_tokens 30000 --max_model_len 32768 --max_num_seqs 12 --tp_size 2
```

一个合并好的 GRPO checkpoint：

```bash
python run_inference.py \
  --model "$CLEF_PROJ/exps/zh-physics/merged_step138_full" \
  --out predictions/no_length_step138.jsonl
```

正式跑之前先加 `--limit 20`，把整条路径走通一遍。

32k 上下文下默认的并发数装不进 KV cache，所以 `--max_num_seqs` 要降下来，同时用 `--tp_size 2` 把 cache 分摊到两张卡。`--max_num_seqs` 调得过低又会让作业跑不完常规的 walltime。2 张卡配 12 条序列是可行的。

## 文件

| 文件 | 作用 |
|---|---|
| `run_inference.py` | 下载测试集 parquet，用 vLLM 贪心解码跑完 1117 题，每题写一行 JSONL，含原始输出。不打分 |
| `evaluate.py` | 套用抽取规则，报总体以及分语言、分学科的准确率、invalid 率、回答长度 |
| `normalize_subject.py` | 32 种学科写法归到 9 类，被 `evaluate.py` 引用 |
| `submit.py` | 提交用的推理。可断点续跑，每批之后按官方格式写一次 `submission.json` |
| `requirements.txt` | 环境冻结 |

`run_inference.py` 只写元数据和原始输出。换一套规则重新打分，只需要再跑 `evaluate.py`，推理不用重做。

`submit.py` 的配置全在文件开头的那组全局变量里。

| 变量 | 含义 |
|---|---|
| `MODEL_PATH` | base 模型目录 |
| `LORA_PATH` | LoRA adapter 目录。留空则不加载 LoRA |
| `MAX_LORA_RANK` | 必须与训练时的 rank 一致（64），否则 vLLM 拒绝加载 adapter |
| `DATA_JSON_PATH` | 数据集 JSON |
| `IMAGE_DIR` | 图片目录 |
| `OUTPUT_JSON_PATH` | 模型原始输出 |
| `SUBMISSION_JSON_PATH` | 提交文件输出 |
| `TEST_MODE` | 设为 `True` 时只跑 `TEST_SAMPLE_COUNT` 条 |

`INVALID` 表示模型用完了推理预算却没有给出最终的 A-E 选项；提交文件的写入逻辑会退回 `"A"`，保证文件始终能通过格式校验。

## 参考

- 任务方仓库 —— [src/baselines](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning/tree/main/2026/src/baselines) · [src/evaluation](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning/tree/main/2026/src/evaluation)
