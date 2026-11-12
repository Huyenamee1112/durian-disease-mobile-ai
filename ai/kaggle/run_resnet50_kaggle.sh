#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
DATA_SOURCE="$ROOT/ai/data/supabase_4class_20261003_balanced110_processed"
DATA_STAGE="$ROOT/ai/kaggle/field_balanced110/data"
KERNEL_DIR="$ROOT/ai/kaggle/resnet50_transformer_field"
OUTPUT_DIR="$ROOT/ai/models/hybrid_resnet50_transformer/field_balanced110_kaggle"

rsync -a --delete "$DATA_SOURCE/" "$DATA_STAGE/"
"$KAGGLE" datasets create -p "$(dirname "$DATA_STAGE")" --dir-mode zip
"$KAGGLE" kernels push -p "$KERNEL_DIR"
echo "Job submitted. Monitor: $KAGGLE kernels status huyentran999/durian-resnet50-transformer-field"
echo "After completion: $KAGGLE kernels output huyentran999/durian-resnet50-transformer-field -p $OUTPUT_DIR"
