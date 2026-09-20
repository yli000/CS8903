#!/usr/bin/env bash
# python 3.12 + CUDA 13 + verl + vLLM nightly + flash-attn 2.8.3 + qwen_vl_utils。
# 下面各步的顺序不能调换。

set -euo pipefail

PROJ="${CLEF_PROJ:?CLEF_PROJ is required (project root to build the venv in)}"
cd "$PROJ"

echo "=== start $(date) on $(hostname) ==="

# 集群上从 module 加载 CUDA 和 python，本地机器跳过
if command -v module >/dev/null 2>&1; then
    module load cuda/13.0.1
    module load python/3.12.5
fi
nvcc --version 2>&1 | tail -2 || true
python3 -V

echo "=== fresh venv ==="
rm -rf venv          # 有作业正在用这个 venv 时不要运行本脚本
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel packaging ninja -q   # 先装 packaging 和 ninja，否则后面有包走源码编译

echo "=== verl + its vLLM extra (pulls torch) ==="
if [ -d "$PROJ/verl" ]; then
    cd "$PROJ/verl" && git pull -q
else
    git clone https://github.com/verl-project/verl.git "$PROJ/verl"
fi
cd "$PROJ/verl"
pip install -e ".[vllm]" -q
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

echo "=== replace vLLM and transformers with nightly ==="
pip uninstall vllm transformers -y          # 上一步装的 release 版没有 Qwen3.5 支持
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly -q

echo "=== flash-attn prebuilt wheel (torch 2.11 + cu13 + cp312) ==="
# ABI 必须与上面装出来的 torch 一致，版本不同就换 URL
FA_URL="https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install "$FA_URL" -q

echo "=== vision utils and helpers ==="
pip install qwen_vl_utils -q
pip install datasets pyarrow pillow pandas wandb peft -q

echo "=== verification ==="
python - <<'PY'
import torch, vllm, flash_attn, transformers, datasets
from flash_attn import flash_attn_func
from transformers import AutoConfig
from vllm.model_executor.models.registry import ModelRegistry
print(f'torch        = {torch.__version__}  cuda={torch.version.cuda}  gpu={torch.cuda.is_available()}')
print(f'vllm         = {vllm.__version__}')
print(f'flash_attn   = {flash_attn.__version__}')
print(f'transformers = {transformers.__version__}')
print(f'datasets     = {datasets.__version__}')
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
print(f'Qwen3.5 model_type={cfg.model_type} arch={cfg.architectures}')
supported = 'Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs()
print(f'vllm supports Qwen3_5 = {supported}')
assert supported, 'vLLM does not list Qwen3_5ForConditionalGeneration; the nightly step did not take effect'
print('=== all checks pass ===')
PY

echo "=== done $(date) ==="
