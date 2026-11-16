#!/usr/bin/env bash
# Build the training image, or run something in it.
#   scripts/docker.sh build
#   scripts/docker.sh test                                         unit tests (no GPU)
#   scripts/docker.sh train recipes/qwen3_8b.json --run NAME [--smoke] [--dry-run]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=email-agent-miles
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
MODE=${1:?Usage: docker.sh build|test|train ...}
shift

if [[ $MODE == build ]]; then
  exec docker build -t "$IMAGE" "$ROOT"
fi

args=(--rm --init --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
      --mount "type=bind,src=$ROOT,dst=$ROOT" --workdir "$ROOT"
      --mount "type=bind,src=$HF_CACHE,dst=$HF_CACHE,readonly" -e "HF_HOME=$HF_CACHE"
      -e "PYTHONPATH=$ROOT:/root/miles:/root/Megatron-LM" -e PYTHONDONTWRITEBYTECODE=1)
# AWS credentials, unless the host has an instance role.
[[ -d $HOME/.aws ]] && args+=(--mount "type=bind,src=$HOME/.aws,dst=/root/.aws,readonly")
for key in AWS_PROFILE AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN MODEL_PATH WANDB_API_KEY; do
  [[ -n ${!key:-} ]] && args+=(-e "$key")
done
# Megatron needs the NVIDIA driver libraries even when it uses no GPU.
no_gpu=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)

case $MODE in
  test) exec docker run "${args[@]}" "${no_gpu[@]}" "$IMAGE" python -m pytest -q -p no:cacheprovider tests/trainer "$@" ;;
  train)
    if [[ " $* " == *" --dry-run "* ]]; then
      exec docker run "${args[@]}" "${no_gpu[@]}" "$IMAGE" python -m email_miles.launch "$@"
    fi
    # Compiled kernels are kept between runs.
    mkdir -p "$ROOT"/.cache/{sglang,torchinductor,triton,tilelang}
    exec docker run "${args[@]}" --gpus all \
      --mount "type=bind,src=$ROOT/.cache/sglang,dst=/root/.cache/sglang" \
      --mount "type=bind,src=$ROOT/.cache/torchinductor,dst=/tmp/torchinductor_root" \
      --mount "type=bind,src=$ROOT/.cache/triton,dst=/root/.triton" \
      --mount "type=bind,src=$ROOT/.cache/tilelang,dst=/root/.tilelang" \
      "$IMAGE" python -m email_miles.launch "$@" ;;
  *) echo "Usage: docker.sh build|test|train ..." >&2; exit 2 ;;
esac
