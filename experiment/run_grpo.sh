#!/usr/bin/env bash
# GRPO 训练。三个实验共用同一份 verl 配置，差异全在下面的 case 块里。
#   bash run_grpo.sh single_subject | tri_subject | no_length_penalty

set -euo pipefail

PROJ="${CLEF_PROJ:?CLEF_PROJ is required (project root holding exps/)}"
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

VARIANT="${1:?usage: run_grpo.sh single_subject|tri_subject|no_length_penalty}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$VARIANT" in
  single_subject)
    DATA_DIR="$PROJ/exps/zh-physics/data"
    REWARD="$HERE/reward_length_penalty.py"
    CKPT_DIR="$PROJ/exps/zh-physics/checkpoints"
    WANDB_NAME="zh-physics-len-cont"
    N_GPUS=8
    ;;
  tri_subject)
    # 唯一改动：训练数据换成 Physics + Chemistry + Biology 三科混合
    DATA_DIR="$PROJ/exps/zh-physchembio/data"
    REWARD="$HERE/reward_length_penalty.py"
    CKPT_DIR="$PROJ/exps/zh-physchembio/checkpoints"
    WANDB_NAME="zh-physchembio-mixed"
    N_GPUS=2
    ;;
  no_length_penalty)
    # 唯一改动：reward 去掉长度惩罚项
    DATA_DIR="$PROJ/exps/zh-physics/data"
    REWARD="$HERE/reward_no_length_penalty.py"
    CKPT_DIR="$PROJ/exps/zh-physics/checkpoints_no_length"
    WANDB_NAME="zh-physics-no-length"
    N_GPUS=8
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 1
    ;;
esac

export HF_HOME="$PROJ/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# reward 函数把每次 rollout 和各项分数追加到这里
export ROLLOUT_LOG_PATH="$CKPT_DIR/../rollout_${VARIANT}.jsonl"
mkdir -p "$(dirname "$ROLLOUT_LOG_PATH")"

echo "=== $VARIANT start $(date) on $(hostname) ==="
echo "  data   $DATA_DIR"
echo "  reward $REWARD"
echo "  ckpt   $CKPT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-9B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=6 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.critic_warmup=0 \
    trainer.project_name=clef2026-grpo \
    trainer.experiment_name="$WANDB_NAME" \
    trainer.logger='["console","wandb"]' \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode=auto

echo "=== $VARIANT end $(date) ==="
