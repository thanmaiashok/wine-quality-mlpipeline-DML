<p align="center"><img src="docs/flow.svg" alt="Animated MLOps Wine Pipeline pipeline: Ingest → Preprocess → Train → Gate → Visualize" width="100%"/></p>

<p align="center"><sub>10-second tour: Ingest → Preprocess → Train → Gate → Visualize</sub></p>

<p align="center"><img src="docs/mc/intro.svg" width="100%" alt="Jenkins-style CI/CD pipeline for machine learning: data ingestion, preprocessing, training, an evaluation gate and visual reporting."/></p>

<p align="center"><img src="docs/mc/features.svg" width="100%" alt="Key features"/></p>

<a id="what-it-does"></a>
<h2><img src="docs/mc/h2-what-it-does.svg" width="100%" alt="What It Does"/></h2>

<p align="center"><img src="docs/mc/t-01.svg" width="100%" alt="Predicts red wine quality (High / Low) from 11 chemical lab measurements - no human taster needed. Pipeline mirrors real DevOps CI/CD stages:"/></p>

```
Ingest → Preprocess → Train → Evaluate → Visualize
  ↓           ↓          ↓        ↓           ↓
Download    Clean &    Random   Gate: acc   6 visual
CSV from    split      Forest   must ≥ 75%  output
UCI Repo    data       100 trees  or FAIL   charts
```

<p align="center"><img src="docs/mc/t-02.svg" width="100%" alt="If evaluation fails -&gt; pipeline stops. Nothing deploys. Same behavior as Jenkins build gate."/></p>

<a id="dataset"></a>
<h2><img src="docs/mc/h2-dataset.svg" width="100%" alt="Dataset"/></h2>

<p align="center"><img src="docs/mc/t-03.svg" width="100%" alt="Property | Value Name | UCI Wine Quality (Red) Source | UC Irvine ML Repository Rows | 1,599 wine samples Features | 11 chemical measurements (alcohol, pH, acidity, etc.) Target | Quality score -&gt; binary: High (&gt;=7) / Low (&lt;7) Format | CSV (auto-downloaded, no manual setup)"/></p>

<p align="center"><a href="https://archive.ics.uci.edu/ml/datasets/wine+quality"><img src="docs/mc/link-01.svg" height="34" alt="UC Irvine ML Repository"/></a></p>

<a id="results"></a>
<h2><img src="docs/mc/h2-results.svg" width="100%" alt="Results"/></h2>

<p align="center"><img src="docs/mc/t-04.svg" width="100%" alt="Metric | Score Accuracy | 94.06% ROC-AUC | 0.955 F1 Score | 0.732 Train size | 1,279 samples Test size | 320 samples Top predictor: alcohol content (importance score: 0.174)"/></p>

<a id="output-plots"></a>
<h3><img src="docs/mc/h3-output-plots.svg" width="100%" alt="Output Plots"/></h3>

<p align="center"><img src="docs/mc/t-05.svg" width="100%" alt="Plot | Description 0_pipeline_report.png | Full pipeline execution dashboard 1_quality_distribution.png | Quality score distribution + class balance 2_feature_correlations.png | Feature correlation heatmap 3_feature_distributions.png | Key features by quality class (box plots) 4_feature_importance.png | Random Forest feature importance ranking 5_confusion_matrix.png | Confusion matrix (275 correct low, 26 correct high) 6_roc_curve.png | ROC curve (AUC = 0.955)"/></p>

<a id="quick-start"></a>
<h2><img src="docs/mc/h2-quick-start.svg" width="100%" alt="Quick Start"/></h2>

<a id="prerequisites"></a>
<h3><img src="docs/mc/h3-prerequisites.svg" width="100%" alt="Prerequisites"/></h3>

<p align="center"><img src="docs/mc/t-06.svg" width="100%" alt="Python 3.9+ venv or any virtual environment"/></p>

<a id="run-locally"></a>
<h3><img src="docs/mc/h3-run-locally.svg" width="100%" alt="Run Locally"/></h3>

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

<p align="center"><img src="docs/mc/t-07.svg" width="100%" alt="Or run individual stages:"/></p>

```bash
cd pipeline
python stage1_ingest.py       # Download dataset
python stage2_preprocess.py   # Clean + split
python stage3_train.py        # Train model
python stage4_test.py         # Evaluate (fails if acc < 0.75)
python stage5_visualize.py    # Generate plots
python generate_report.py     # Generate report chart + markdown
```

<a id="run-tests"></a>
<h3><img src="docs/mc/h3-run-tests.svg" width="100%" alt="Run Tests"/></h3>

```bash
source venv/bin/activate
pytest tests/ -v
```

<a id="project-structure"></a>
<h2><img src="docs/mc/h2-project-structure.svg" width="100%" alt="Project Structure"/></h2>

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

<a id="jenkins-setup"></a>
<h2><img src="docs/mc/h2-jenkins-setup.svg" width="100%" alt="Jenkins Setup"/></h2>

<p align="center"><img src="docs/mc/t-08.svg" width="100%" alt="Create new Pipeline job in Jenkins Point to this repo Jenkins reads Jenkinsfile automatically Pipeline stages map 1:1 to Python scripts Build artifacts: models/rf_model.pkl + all plots Stages in Jenkinsfile: Checkout -&gt; pull latest code Build -&gt; install requirements.txt Data Ingestion -&gt; stage1_ingest.py Preprocessing -&gt; stage2_preprocess.py Unit Tests -&gt; pytest tests/ Model Training -&gt; stage3_train.py Model Evaluation -&gt; stage4_test.py &lt;- pipeline fails here if accuracy &lt; 0.75 Package -&gt; stage5_visualize.py + archive artifacts Deploy -&gt; publish plots as build artifacts"/></p>

<a id="tech-stack"></a>
<h2><img src="docs/mc/h2-tech-stack.svg" width="100%" alt="Tech Stack"/></h2>

<p align="center"><img src="docs/mc/t-09.svg" width="100%" alt="Tool | Purpose Python 3.9+ | Core language scikit-learn | Random Forest, metrics, preprocessing pandas / numpy | Data manipulation matplotlib / seaborn | Visualizations pytest | Unit testing joblib | Model serialization Jenkins | CI/CD orchestration"/></p>

<a id="license"></a>
<h2><img src="docs/mc/h2-license.svg" width="100%" alt="License"/></h2>

<p align="center"><img src="docs/mc/t-10.svg" width="100%" alt="MIT - free to use, modify, distribute."/></p>

<p align="center"><a href="https://github.com/thanmaiashok"><img src="docs/mc/footer.svg" width="100%" alt="Built by Thanmai A, founder of FoxynAI"/></a></p>
