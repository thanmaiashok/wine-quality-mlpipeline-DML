# MLOps Wine Pipeline

<p align="center"><img src="docs/flow.svg" alt="Animated MLOps Wine Pipeline pipeline: Ingest → Preprocess → Train → Gate → Visualize" width="100%"/></p>
<p align="center"><sub>10-second tour: Ingest → Preprocess → Train → Gate → Visualize</sub></p>

> **Jenkins-style CI/CD pipeline for machine learning** — automated data ingestion, preprocessing, model training, evaluation gate, and visual reporting. Built with Python + scikit-learn, deployable via Jenkins or run locally in one command.

---

## What It Does

Predicts **red wine quality** (High / Low) from 11 chemical lab measurements — no human taster needed.

Pipeline mirrors real DevOps CI/CD stages:

```
Ingest → Preprocess → Train → Evaluate → Visualize
  ↓           ↓          ↓        ↓           ↓
Download    Clean &    Random   Gate: acc   6 visual
CSV from    split      Forest   must ≥ 75%  output
UCI Repo    data       100 trees  or FAIL   charts
```

If evaluation fails → pipeline stops. Nothing deploys. Same behavior as Jenkins build gate.

---

## Dataset

| Property | Value |
|----------|-------|
| Name | UCI Wine Quality (Red) |
| Source | [UC Irvine ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| Rows | 1,599 wine samples |
| Features | 11 chemical measurements (alcohol, pH, acidity, etc.) |
| Target | Quality score → binary: High (≥7) / Low (<7) |
| Format | CSV (auto-downloaded, no manual setup) |

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **94.06%** |
| ROC-AUC | **0.955** |
| F1 Score | 0.732 |
| Train size | 1,279 samples |
| Test size | 320 samples |

Top predictor: `alcohol` content (importance score: 0.174)

### Output Plots

| Plot | Description |
|------|-------------|
| `0_pipeline_report.png` | Full pipeline execution dashboard |
| `1_quality_distribution.png` | Quality score distribution + class balance |
| `2_feature_correlations.png` | Feature correlation heatmap |
| `3_feature_distributions.png` | Key features by quality class (box plots) |
| `4_feature_importance.png` | Random Forest feature importance ranking |
| `5_confusion_matrix.png` | Confusion matrix (275 correct low, 26 correct high) |
| `6_roc_curve.png` | ROC curve (AUC = 0.955) |

---

## Quick Start

### Prerequisites
- Python 3.9+
- `venv` or any virtual environment

### Run Locally

```bash
# 1. Clone repo
git clone https://github.com/thanmaiashok/wine-quality-mlpipeline-DML-.git
cd wine-quality-mlpipeline-DML-

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Or run individual stages:

```bash
cd pipeline
python stage1_ingest.py       # Download dataset
python stage2_preprocess.py   # Clean + split
python stage3_train.py        # Train model
python stage4_test.py         # Evaluate (fails if acc < 0.75)
python stage5_visualize.py    # Generate plots
python generate_report.py     # Generate report chart + markdown
```

### Run Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## Project Structure

```
wine-quality-mlpipeline-DML-/
├── Jenkinsfile                  # Jenkins pipeline definition
├── README.md
├── requirements.txt
├── run_pipeline.sh              # One-command local runner
│
├── pipeline/
│   ├── pipeline_runner.py       # Main entry point (runs all stages)
│   ├── stage1_ingest.py         # Stage 1: Download CSV from UCI
│   ├── stage2_preprocess.py     # Stage 2: Clean, scale, split
│   ├── stage3_train.py          # Stage 3: Train Random Forest
│   ├── stage4_test.py           # Stage 4: Evaluate + pipeline gate
│   ├── stage5_visualize.py      # Stage 5: Generate 6 plots
│   └── generate_report.py       # Generate summary dashboard
│
├── data/
│   ├── winedataset.md           # Dataset documentation
│   ├── raw/                     # Auto-downloaded CSV (git-ignored)
│   └── processed/               # Train/test splits (git-ignored)
│
├── models/                      # Saved model + metrics (git-ignored)
│
├── outputs/
│   ├── pipeline_report.md       # Execution report
│   └── plots/                   # 7 output PNG charts
│
└── tests/
    └── test_pipeline.py         # Unit tests (pytest)
```

---

## Jenkins Setup

1. Create new Pipeline job in Jenkins
2. Point to this repo
3. Jenkins reads `Jenkinsfile` automatically
4. Pipeline stages map 1:1 to Python scripts
5. Build artifacts: `models/rf_model.pkl` + all plots

Stages in Jenkinsfile:
- **Checkout** → pull latest code
- **Build** → install `requirements.txt`
- **Data Ingestion** → `stage1_ingest.py`
- **Preprocessing** → `stage2_preprocess.py`
- **Unit Tests** → `pytest tests/`
- **Model Training** → `stage3_train.py`
- **Model Evaluation** → `stage4_test.py` ← pipeline fails here if accuracy < 0.75
- **Package** → `stage5_visualize.py` + archive artifacts
- **Deploy** → publish plots as build artifacts

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| scikit-learn | Random Forest, metrics, preprocessing |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualizations |
| pytest | Unit testing |
| joblib | Model serialization |
| Jenkins | CI/CD orchestration |

---

## License

MIT — free to use, modify, distribute.
