#!/usr/bin/env bash
# One-time setup for petitellm-distillation:
# 1. Install deps
# 2. Download FineWeb sp1024 tokenizer + 2 train shards + val shard from willdepueoai/parameter-golf
# 3. Cache teacher top-k logits:
#    - If TEACHER_LOGITS_HF_REPO is set and resolvable, snapshot_download from there.
#    - Else, run generate_teacher.py to train a modest self-distillation teacher and cache.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python3}"

echo "[1/4] Installing requirements ..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -r requirements.txt
else
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
fi

mkdir -p data/datasets/fineweb10B_sp1024 data/tokenizers data/teacher_logits

DATA_DIR="data/datasets/fineweb10B_sp1024"
TOKENIZER="data/tokenizers/fineweb_1024_bpe.model"

need_download=false
[ -f "$TOKENIZER" ] || need_download=true
ls "$DATA_DIR"/fineweb_val_*.bin >/dev/null 2>&1 || need_download=true
TRAIN_COUNT=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRAIN_COUNT" -lt 2 ]; then
    need_download=true
fi

if [ "$need_download" = true ]; then
    echo "[2/4] Downloading FineWeb shards + tokenizer from willdepueoai/parameter-golf ..."
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='willdepueoai/parameter-golf',
    repo_type='dataset',
    local_dir='./data',
    allow_patterns=[
        'datasets/fineweb10B_sp1024/fineweb_val_*',
        'datasets/fineweb10B_sp1024/fineweb_train_000.bin',
        'datasets/fineweb10B_sp1024/fineweb_train_001.bin',
        'tokenizers/fineweb_1024_bpe.model',
    ],
)
print('Download complete.')
"
else
    echo "[2/4] FineWeb shards already present, skipping download."
fi

# Verify
if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: tokenizer not found after download at $TOKENIZER" >&2
    exit 1
fi
TRAIN_COUNT=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l | tr -d ' ')
VAL_COUNT=$(ls "$DATA_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "      train shards: $TRAIN_COUNT  val shards: $VAL_COUNT"
if [ "$TRAIN_COUNT" -lt 2 ] || [ "$VAL_COUNT" -lt 1 ]; then
    echo "ERROR: expected >=2 train shards and >=1 val shard" >&2
    exit 1
fi

# Teacher logits
TEACHER_INDICES="data/teacher_logits/teacher_topk_indices.bin"
TEACHER_LOGPROBS="data/teacher_logits/teacher_topk_logprobs.bin"
TEACHER_META="data/teacher_logits/meta.json"

if [ -f "$TEACHER_INDICES" ] && [ -f "$TEACHER_LOGPROBS" ] && [ -f "$TEACHER_META" ]; then
    echo "[3/4] Teacher logits already cached, skipping."
elif [ -n "${TEACHER_LOGITS_HF_REPO:-}" ]; then
    echo "[3/4] Downloading teacher logits from HF repo: $TEACHER_LOGITS_HF_REPO ..."
    "$PY" -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ['TEACHER_LOGITS_HF_REPO'],
    repo_type='dataset',
    local_dir='./data/teacher_logits',
)
"
    if [ ! -f "$TEACHER_INDICES" ] || [ ! -f "$TEACHER_LOGPROBS" ] || [ ! -f "$TEACHER_META" ]; then
        echo "ERROR: downloaded HF teacher repo missing required files" >&2
        exit 1
    fi
else
    echo "[3/4] Running self-distillation teacher: generate_teacher.py ..."
    "$PY" generate_teacher.py || {
        echo "ERROR: generate_teacher.py failed. Cleaning up partial files." >&2
        rm -f "$TEACHER_INDICES" "$TEACHER_LOGPROBS" "$TEACHER_META"
        exit 1
    }
fi

# Report sizes
echo "[4/4] Summary:"
du -sh "$DATA_DIR" "$TOKENIZER" "data/teacher_logits" 2>/dev/null || true
echo "Done."
