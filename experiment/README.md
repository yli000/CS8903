English | [中文](README.zh-CN.md)

# Experiment

GRPO fine-tuning of Qwen3.5-9B with [veRL](https://github.com/verl-project/verl), using LoRA adapters. The reward combines answer correctness, a response-length penalty, and a format bonus.

Three runs. Each changes exactly one thing against the first, and all three are reported at training step 138.

| Run | What changes | Training data |
|---|---|---|
| `single_subject` | reference configuration | Chinese Physics, 371 train / 87 val |
| `tri_subject` | training data | Chinese Physics + Chemistry + Biology, 927 (371/328/228) / 316 (87/175/54) |
| `no_length_penalty` | reward, length term removed | Chinese Physics, 371 / 87 |

"val" is veRL's in-training validation set, evaluated every 5 steps to track progress. It never enters the reward or a gradient update.

## Reward

`c` is correctness, `f` is a closed `</think>`, both in {0,1}. `ℓ` is the response length in characters. `L = 6144`.

```
reward_length_penalty.py:      R = c − 0.3 · (1 − c) · min(ℓ/L, 1) + 0.05 · f
reward_no_length_penalty.py:   R = c + 0.05 · f
```

The length penalty applies to wrong answers only, through the `(1 − c)` factor. A long response that reaches the right letter is not penalized.

The format bonus is 0.05 against a correctness term of 1.0. Emitting `</think>` around a wrong answer gains 0.05 and forfeits 1.0.

veRL calls `compute_score(data_source, solution_str, ground_truth, extra_info)`. Each call appends the rollout, the extracted letter, the score and the length to a JSONL trace at `$ROLLOUT_LOG_PATH`.

## Training config

Identical across the three runs. Every value is a flag in [`run_grpo.sh`](run_grpo.sh).

| | |
|---|---|
| Base model | Qwen3.5-9B |
| Algorithm | GRPO, `adv_estimator=grpo` |
| Adapter | LoRA rank = α = 64, `all-linear` |
| Actor / reference | FSDP2, gradient checkpointing, no offload |
| Rollout | vLLM, n = 4 per prompt, T = 1.0, `max_model_len` 8192 |
| Optimizer | lr 5e-6 |
| KL | `use_kl_loss=True`, coef 0.01, `low_var_kl`, not in reward |
| Batching | train batch 16, PPO mini-batch 16, micro-batch 1 per GPU |
| Lengths | max prompt 1024, max response 6144 |
| Steps | 138 |

## Environment

Install veRL, the vLLM nightly, Flash Attention, and the Qwen-VL vision utilities. The five steps below are what [`setup_env.sh`](setup_env.sh) runs in one go; run the script on a cluster, or follow the steps by hand.

### 1. Create the venv and install veRL

On a cluster, load CUDA and python from modules first: `module load cuda/13.0.1 python/3.12.5`.

```bash
export CLEF_PROJ=/path/to/project-root   # venv, veRL and exps/ all live here
cd "$CLEF_PROJ"
python3 -m venv venv && source venv/bin/activate

# packaging and ninja go first, or later packages fall back to a source build
pip install --upgrade pip setuptools wheel packaging ninja

git clone https://github.com/verl-project/verl.git
cd verl
pip install -e ".[vllm]"   # installs veRL together with its vLLM extra
```

### 2. Replace vLLM with the nightly

The previous step installs a release version of vLLM. Qwen3.5 support was not in a release yet, so it has to be swapped for a nightly wheel.

```bash
pip uninstall vllm transformers -y
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

### 3. Install Flash Attention matched to your torch and CUDA

First check the versions the previous steps installed:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Then pick the prebuilt wheel matching those versions. The URL below is for **torch 2.11 + CUDA 13 + Python 3.12** — change it if your versions differ. Building from source against a different ABI fails at import time.

```bash
pip install "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### 4. Install the vision utilities and helpers

```bash
pip install qwen_vl_utils   # vision-side preprocessing for the Qwen-VL family
pip install datasets pyarrow pillow pandas wandb peft
```

### 5. Check that vLLM sees Qwen3.5

```bash
python -c "from vllm.model_executor.models.registry import ModelRegistry; \
print('Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
```

`False` means step 2 did not take effect. `setup_env.sh` asserts on this check, so a failed environment build stops here instead of failing hours into training.

### Credentials

```bash
export HF_TOKEN=...          # the two datasets are gated
export WANDB_API_KEY=...     # training only
```

`setup_env.sh` deletes and rebuilds the venv at `$CLEF_PROJ/venv`. Do not run it while a job is using that venv.

## Run

8× H100-80GB or 8× H200 for `single_subject` and `no_length_penalty`, about 5.5 hours for 138 steps. `tri_subject` ran on 2× A100-80GB at 7-8 minutes per step.

Without the length penalty the rollouts stay long, and one step on 2 GPUs takes about 2.5 hours. A100-40GB does not hold 9B + rank-64 LoRA + 6144 response length.

```bash
# 1. data
python prep_data.py --subject Physics \
  --output_dir "$CLEF_PROJ/exps/zh-physics/data"
python prep_data.py --subjects Physics,Chemistry,Biology \
  --output_dir "$CLEF_PROJ/exps/zh-physchembio/data"

# 2. training
bash run_grpo.sh single_subject
bash run_grpo.sh tri_subject
bash run_grpo.sh no_length_penalty

# 3. merge a checkpoint into a model vLLM can load
python merge_checkpoint.py \
  --ckpt_dir "$CLEF_PROJ/exps/zh-physics/checkpoints_no_length/global_step_138/actor" \
  --base_dir "$CLEF_PROJ/exps/zh-physics/merged_step138" \
  --full_dir "$CLEF_PROJ/exps/zh-physics/merged_step138_full"

# 4. score it with the baseline scripts
cd ../baseline
python run_inference.py --model "$CLEF_PROJ/exps/zh-physics/merged_step138_full" \
  --out predictions/no_length_step138.jsonl
python evaluate.py --predictions predictions/no_length_step138.jsonl
```

On a SLURM cluster, wrap each command in an `sbatch` script with the partition, GPU count and walltime for your allocation. veRL is launched with plain `bash`, so nothing here depends on a scheduler.

`--output_dir` on `prep_data.py` is required. A default value makes it easy to overwrite one experiment's parquet while another job is reading it.

Checkpoints go to `$CLEF_PROJ/exps/<run>/checkpoints*/global_step_<N>/actor`. `save_freq` and `max_actor_ckpt_to_keep` in `run_grpo.sh` decide which ones stay on disk, which is what limits the steps available for evaluation later. Both merge stages are idempotent, so a job killed at the walltime can be resubmitted.

## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | Builds the training environment, verifies Qwen3.5 support in vLLM |
| `prep_data.py` | EXAMS-V → veRL parquet. Language and subject filtering, answer-key normalization over 23 surface forms, cross-language subject canonicalization, JPEG re-encoding capped at 1024 px |
| `reward_length_penalty.py` | Reward for `single_subject` and `tri_subject` |
| `reward_no_length_penalty.py` | Same reward, length term removed |
| `run_grpo.sh` | One veRL invocation for all three runs. The `case` block at the top holds everything that differs |
| `merge_checkpoint.py` | FSDP shards → HuggingFace base + LoRA adapter → one merged model |

The prompt is built in `prep_data.py` and is identical at training and evaluation time:

```
<image>\n分析这道题的题目、所有选项和存在的视觉元素（如有），仅回答一个选项：A、B、C、D或E。
```

## References

- veRL — [docs](https://verl.readthedocs.io/en/latest/start/quickstart.html) · [data prep and training configs](https://github.com/verl-project/verl/tree/main/examples) · [reward score implementations](https://github.com/verl-project/verl/tree/main/verl/utils/reward_score)
- `flash-attn` prebuilt wheels — [adithyaxx/flash-attention](https://github.com/adithyaxx/flash-attention/releases)
