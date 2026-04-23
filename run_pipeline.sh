#!/bin/bash
# Local pipeline runner — mirrors Jenkinsfile stages without Jenkins
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  CI/CD ML PIPELINE — Local Execution"
echo "=============================================="

echo ""
echo "[BUILD] Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt -q

echo ""
echo "[INGEST] Fetching dataset..."
cd pipeline && python stage1_ingest.py && cd ..

echo ""
echo "[PREPROCESS] Processing data..."
cd pipeline && python stage2_preprocess.py && cd ..

echo ""
echo "[TEST] Running unit tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "[TRAIN] Training model..."
cd pipeline && python stage3_train.py && cd ..

echo ""
echo "[EVALUATE] Evaluating model..."
cd pipeline && python stage4_test.py && cd ..

echo ""
echo "[VISUALIZE] Generating plots..."
cd pipeline && python stage5_visualize.py && cd ..

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE"
echo "  Results: outputs/plots/"
echo "=============================================="
