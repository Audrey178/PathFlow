#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/prepare_data.sh --split <split> [--data_dir <dir>] [--max_workers <n>]

Examples:
  bash scripts/prepare_data.sh --split train
  bash scripts/prepare_data.sh --data_dir datasets --split val --max_workers 4

Options:
  --split <split>       Required. CSV split name, e.g. train, val, test.
  --data_dir <dir>      Dataset root directory. Default: datasets
  --max_workers <n>     Number of workers for frame extraction. Default: 4
  -h, --help            Show this help message.
EOF
}

DATA_DIR="datasets"
SPLIT=""
MAX_WORKERS="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --split requires a value." >&2
                usage >&2
                exit 1
            fi
            SPLIT="$2"
            shift 2
            ;;
        --data_dir)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --data_dir requires a value." >&2
                usage >&2
                exit 1
            fi
            DATA_DIR="$2"
            shift 2
            ;;
        --max_workers)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --max_workers requires a value." >&2
                usage >&2
                exit 1
            fi
            MAX_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$SPLIT" ]]; then
    echo "Error: --split is required." >&2
    usage >&2
    exit 1
fi

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max_workers must be a positive integer." >&2
    exit 1
fi

VIDEO_DIR="$DATA_DIR/videos"
FRAME_DIR="$DATA_DIR/frames"
CSV_PATH="$DATA_DIR/csv/$SPLIT.csv"
CSV_BACKUP_PATH="$CSV_PATH.bak"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is required but was not found in PATH." >&2
    exit 1
fi

if [[ ! -d "$VIDEO_DIR" ]]; then
    echo "Error: video directory not found: $VIDEO_DIR" >&2
    exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: split CSV not found: $CSV_PATH" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" && ! -f ".env" ]]; then
    echo "Error: HF_TOKEN is not set and .env was not found." >&2
    echo "Feature extraction requires access to HuggingFace model MahmoodLab/UNI2-h." >&2
    exit 1
fi

echo "Preparing data"
echo "  repo root:    $REPO_ROOT"
echo "  data dir:     $DATA_DIR"
echo "  split:        $SPLIT"
echo "  videos:       $VIDEO_DIR"
echo "  frames:       $FRAME_DIR"
echo "  csv:          $CSV_PATH"
echo "  max workers:  $MAX_WORKERS"
echo

echo "Step 1/3: extracting frames from videos..."
python preprocessing_video.py \
    --video_dir "$VIDEO_DIR" \
    --frame_dir "$FRAME_DIR" \
    --max_workers "$MAX_WORKERS"

echo
echo "Backing up CSV before optical-flow frame selection..."
cp "$CSV_PATH" "$CSV_BACKUP_PATH"
echo "Backup written to: $CSV_BACKUP_PATH"

echo
echo "Step 2/3: selecting frames with optical flow..."
python utils/finding_frame_idx_OF.py \
    --csv_path "$CSV_PATH"

echo
echo "Step 3/3: extracting UNI2-h features..."
python utils/feat_extract.py \
    --data_dir "$DATA_DIR" \
    --split "$SPLIT"

echo
echo "Data preparation finished for split: $SPLIT"
