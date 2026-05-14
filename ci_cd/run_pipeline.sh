#!/bin/bash
# Local pipeline runner — mirrors Jenkinsfile stages without Jenkins
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "=============================================="
echo "  CI/CD ML PIPELINE — Local Execution"
echo "=============================================="

echo ""
echo "[BUILD] Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt -q

echo ""
echo "[INGEST] Fetching dataset..."
cd etl && python stage1_ingest.py && cd "$ROOT_DIR"

echo ""
echo "[PREPROCESS] Processing data..."
cd etl && python stage2_preprocess.py && cd "$ROOT_DIR"

echo ""
echo "[TEST] Running unit tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "[TRAIN] Training model..."
cd pipeline && python stage3_train.py && cd "$ROOT_DIR"

echo ""
echo "[EVALUATE] Evaluating model..."
cd pipeline && python stage4_test.py && cd "$ROOT_DIR"

echo ""
echo "[VISUALIZE] Generating plots..."
cd pipeline && python stage5_visualize.py && cd "$ROOT_DIR"

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE"
echo "  Results: outputs/plots/"
echo "=============================================="
