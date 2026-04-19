#!/usr/bin/env bash
# Evaluate train.py: run training + compression round-trip, parse val_bpb.
# Always outputs a parseable summary block.
set -uo pipefail

cd "$(dirname "$0")/.."

summary() {
    local val_bpb="${1:-ERROR}"
    local artifact_bytes="${2:-0}"
    local line_count="${3:-0}"
    local valid="${4:-false}"
    echo "---"
    printf "val_bpb:          %s\n" "$val_bpb"
    printf "artifact_bytes:   %s\n" "$artifact_bytes"
    printf "line_count:       %s\n" "$line_count"
    printf "valid:            %s\n" "$valid"
}

# --- Pre-flight checks ---

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found." >&2
    summary "ERROR" "0" "0" "false"
    exit 0
fi

LINE_COUNT=$(wc -l < train.py | tr -d ' ')

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "ERROR: CUDA not available." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"
    exit 0
fi

DATA_DIR="data/datasets/fineweb10B_sp1024"
TOKENIZER="data/tokenizers/fineweb_1024_bpe.model"
TEACHER_IDX="data/teacher_logits/teacher_topk_indices.bin"
TEACHER_LP="data/teacher_logits/teacher_topk_logprobs.bin"
TEACHER_META="data/teacher_logits/meta.json"

for f in "$TOKENIZER" "$TEACHER_IDX" "$TEACHER_LP" "$TEACHER_META"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: missing $f. Run: bash prepare.sh" >&2
        summary "ERROR" "0" "$LINE_COUNT" "false"
        exit 0
    fi
done

# --- Run training (10 min cap + 2 min buffer for compression + eval round-trip) ---

TMPLOG=$(mktemp)
trap 'rm -f "$TMPLOG"' EXIT

TRAIN_EXIT=0
timeout 720 python3 train.py 2>&1 | tee "$TMPLOG" >&2 || TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -ne 0 ] && [ "$TRAIN_EXIT" -ne 124 ]; then
    echo "ERROR: training exited with code $TRAIN_EXIT." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"
    exit 0
fi

# --- Parse results ---
# Contract: train.py prints a line of the form:
#   final_int8_zlib_roundtrip_exact val_bpb:<float>
# and a line:
#   Total submission size: <N> bytes

VAL_BPB=$(grep -oE 'final_int8_zlib_roundtrip_exact[^:]*val_bpb:[0-9]+\.[0-9]+' "$TMPLOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
ARTIFACT_BYTES=$(grep -oE 'Total submission size:[[:space:]]*[0-9]+' "$TMPLOG" | tail -1 | grep -oE '[0-9]+$')

if [ -z "$VAL_BPB" ]; then
    echo "ERROR: could not parse val_bpb from training output." >&2
    summary "ERROR" "${ARTIFACT_BYTES:-0}" "$LINE_COUNT" "false"
    exit 0
fi

if [ -z "$ARTIFACT_BYTES" ]; then
    echo "ERROR: could not parse artifact size from training output." >&2
    summary "$VAL_BPB" "0" "$LINE_COUNT" "false"
    exit 0
fi

VALID="true"
if [ "$ARTIFACT_BYTES" -gt 16000000 ]; then
    echo "WARNING: artifact_bytes $ARTIFACT_BYTES exceeds 16,000,000 byte limit." >&2
    VALID="false"
fi

summary "$VAL_BPB" "$ARTIFACT_BYTES" "$LINE_COUNT" "$VALID"
