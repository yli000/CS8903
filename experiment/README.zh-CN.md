[English](README.md) | 中文

# Experiment

用 [veRL](https://github.com/verl-project/verl) 对 Qwen3.5-9B 做 GRPO 微调，走 LoRA adapter。reward 由答案正确性、回答长度惩罚、格式奖励三项组成。

三个实验：每个相对第一个只改一个地方，三个都取训练第 138 步的结果。

| 实验 | 改动 | 训练数据 |
|---|---|---|
| `single_subject` | 基准配置 | 中文物理，371 训练 / 87 验证 |
| `tri_subject` | 训练数据 | 中文物理 + 化学 + 生物，927（371/328/228）/ 316（87/175/54） |
| `no_length_penalty` | reward，去掉长度项 | 中文物理，371 / 87 |

这里的验证集是 veRL 训练过程中的验证集，每 5 步评一次用来观察进展。它不参与 reward 计算，也不参与梯度更新。

## Reward

`c` 表示是否答对，`f` 表示是否闭合 `</think>`，取值都是 0 或 1。`ℓ` 是回答的字符数。`L = 6144`。

```
reward_length_penalty.py:      R = c − 0.3 · (1 − c) · min(ℓ/L, 1) + 0.05 · f
reward_no_length_penalty.py:   R = c + 0.05 · f
```

长度惩罚通过 `(1 − c)` 这一项只作用在答错的回答上。写得长但答对的回答不扣分。

格式奖励是 0.05，正确性那一项是 1.0。给一个错答案加上 `</think>` 得到 0.05，同时失去 1.0。

veRL 调用的入口是 `compute_score(data_source, solution_str, ground_truth, extra_info)`。每次调用都会把 rollout、抽出的字母、分数和长度追加写入 `$ROLLOUT_LOG_PATH` 指向的 JSONL。

## 训练配置

三个实验完全相同。每一项都是 [`run_grpo.sh`](run_grpo.sh) 里的一个 flag。

| | |
|---|---|
| 基础模型 | Qwen3.5-9B |
| 算法 | GRPO，`adv_estimator=grpo` |
| Adapter | LoRA rank = α = 64，`all-linear` |
| Actor 与参考策略 | FSDP2，gradient checkpointing，不做 offload |
| Rollout | vLLM，每个 prompt n = 4，T = 1.0，`max_model_len` 8192 |
| 优化器 | lr 5e-6 |
| KL | `use_kl_loss=True`，系数 0.01，`low_var_kl`，不计入 reward |
| 批量 | train batch 16，PPO mini-batch 16，每卡 micro-batch 1 |
| 长度 | max prompt 1024，max response 6144 |
| 步数 | 138 |

## 环境

安装 veRL、vLLM nightly、Flash Attention 和 Qwen-VL 的视觉工具。下面五步就是 [`setup_env.sh`](setup_env.sh) 一次性做完的事情。在集群上直接跑脚本，也可以照着这几步手动执行。

### 1. 建 venv 并安装 veRL

在集群上，这一步之前先从 module 加载 CUDA 和 python：`module load cuda/13.0.1 python/3.12.5`。

```bash
export CLEF_PROJ=/path/to/project-root   # venv、veRL 源码和 exps/ 都放在这个目录下
cd "$CLEF_PROJ"
python3 -m venv venv && source venv/bin/activate

# packaging 和 ninja 要先装，否则后面有包会转去从源码编译
pip install --upgrade pip setuptools wheel packaging ninja

git clone https://github.com/verl-project/verl.git
cd verl
pip install -e ".[vllm]"   # 安装 veRL 以及它的 vLLM extra
```

### 2. 把 vLLM 换成 nightly

上一步装的是 release 版 vLLM。当时 release 里还没有 Qwen3.5 的支持，所以要换成 nightly 的 wheel。

```bash
pip uninstall vllm transformers -y
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

### 3. 安装与 torch、CUDA 匹配的 Flash Attention

先看一下前面两步装成了什么版本：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

再挑一个与这些版本对应的预编译 wheel。下面这个地址对应 **torch 2.11 + CUDA 13 + Python 3.12**，版本不同就换掉。从源码编译时 ABI 不匹配会在 import 阶段失败。

```bash
pip install "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### 4. 安装视觉工具和其他依赖

```bash
pip install qwen_vl_utils   # Qwen-VL 系列的视觉侧预处理
pip install datasets pyarrow pillow pandas wandb peft
```

### 5. 确认 vLLM 认得 Qwen3.5

```bash
python -c "from vllm.model_executor.models.registry import ModelRegistry; \
print('Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
```

输出 `False` 说明第 2 步没有生效。`setup_env.sh` 在这里加了断言，环境没装对会当场停下，而不是等训练跑了几个小时才暴露。

### 凭据

```bash
export HF_TOKEN=...          # 两个数据集需要权限
export WANDB_API_KEY=...     # 只有训练需要
```

`setup_env.sh` 会删掉 `$CLEF_PROJ/venv` 并重建。有作业正在使用这个 venv 的时候不要运行它。

## 运行

`single_subject` 和 `no_length_penalty` 用 8 张 H100-80GB 或 8 张 H200，138 步约 5.5 小时。`tri_subject` 在 2 张 A100-80GB 上跑，每步 7 到 8 分钟。

去掉长度惩罚以后 rollout 一直很长，2 张卡单步要 2.5 小时左右。A100-40GB 装不下 9B 加 rank-64 LoRA 加 6144 的 response length。

```bash
# 1. 数据
python prep_data.py --subject Physics \
  --output_dir "$CLEF_PROJ/exps/zh-physics/data"
python prep_data.py --subjects Physics,Chemistry,Biology \
  --output_dir "$CLEF_PROJ/exps/zh-physchembio/data"

# 2. 训练
bash run_grpo.sh single_subject
bash run_grpo.sh tri_subject
bash run_grpo.sh no_length_penalty

# 3. 把 checkpoint 合并成 vLLM 能加载的模型
python merge_checkpoint.py \
  --ckpt_dir "$CLEF_PROJ/exps/zh-physics/checkpoints_no_length/global_step_138/actor" \
  --base_dir "$CLEF_PROJ/exps/zh-physics/merged_step138" \
  --full_dir "$CLEF_PROJ/exps/zh-physics/merged_step138_full"

# 4. 用 baseline 的脚本打分
cd ../baseline
python run_inference.py --model "$CLEF_PROJ/exps/zh-physics/merged_step138_full" \
  --out predictions/no_length_step138.jsonl
python evaluate.py --predictions predictions/no_length_step138.jsonl
```

在 SLURM 集群上，把每条命令套进一个 `sbatch` 脚本，按自己的账户填分区、卡数和 walltime。veRL 本身用普通 `bash` 启动，这里没有任何东西依赖调度器。

`prep_data.py` 的 `--output_dir` 是必填的。带默认值的时候很容易在另一个作业正在读 parquet 的情况下把它覆盖掉。

checkpoint 写到 `$CLEF_PROJ/exps/<run>/checkpoints*/global_step_<N>/actor`。`run_grpo.sh` 里的 `save_freq` 和 `max_actor_ckpt_to_keep` 决定哪些会留在磁盘上，后面能评测哪些步数就受此限制。两个合并阶段都是幂等的，作业被 walltime 中断后重新提交即可。

## 文件

| 文件 | 作用 |
|---|---|
| `setup_env.sh` | 建训练环境，验证 vLLM 支持 Qwen3.5 |
| `prep_data.py` | EXAMS-V 转成 veRL 的 parquet。语言与学科筛选、23 种 answer_key 写法归一、跨语言学科名归一、图片重新编码为 JPEG 且长边不超过 1024 px |
| `reward_length_penalty.py` | `single_subject` 和 `tri_subject` 用的 reward |
| `reward_no_length_penalty.py` | 同一个 reward，去掉长度项 |
| `run_grpo.sh` | 三个实验共用一次 veRL 调用，开头的 `case` 块里是全部差异 |
| `merge_checkpoint.py` | FSDP 分片 → HuggingFace base 加 LoRA adapter → 一个合并好的模型 |

prompt 在 `prep_data.py` 里构造，训练和评测时完全一致：

```
<image>\n分析这道题的题目、所有选项和存在的视觉元素（如有），仅回答一个选项：A、B、C、D或E。
```

## 参考

- veRL —— [文档](https://verl.readthedocs.io/en/latest/start/quickstart.html) · [数据准备与训练配置](https://github.com/verl-project/verl/tree/main/examples) · [reward 实现](https://github.com/verl-project/verl/tree/main/verl/utils/reward_score)
- `flash-attn` 预编译 wheel —— [adithyaxx/flash-attention](https://github.com/adithyaxx/flash-attention/releases)
