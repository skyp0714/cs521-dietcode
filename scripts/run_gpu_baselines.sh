#!/usr/bin/env bash
# Run serially so the three baselines do not compete for the same GPU.
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"
benchmark_python="${DIETCODE_PYTHON:-$repository_root/.venv-gpu/bin/python}"
gpu_work_root="${DIETCODE_GPU_ROOT:-$repository_root/.cache/gpu}"
output_directory="${DIETCODE_RESULTS_DIR:-$repository_root/results/gpu}"
mkdir -p "$gpu_work_root/tmp" "$output_directory"
export TMPDIR="$gpu_work_root/tmp"

"$benchmark_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$benchmark_python" -m pip freeze --local > "$output_directory/environment.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$output_directory/gpu.csv"
git rev-parse HEAD > "$output_directory/source-commit.txt"
git status --porcelain > "$output_directory/source-status.txt"
git diff HEAD -- src/dietcode configs > "$output_directory/source.patch"
cp configs/workloads.json "$output_directory/workloads.json"

for backend in eager inductor inductor-max-autotune; do
    # Fresh cache directories for this backend in this invocation.
    backend_cache="$(mktemp -d "$gpu_work_root/$backend.XXXXXXXX")"
    TORCHINDUCTOR_CACHE_DIR="$backend_cache/inductor" \
    TRITON_CACHE_DIR="$backend_cache/triton" \
        "$benchmark_python" -m dietcode --backend "$backend" --device cuda \
        --dtype float16 --output "$output_directory/$backend.json"
done
