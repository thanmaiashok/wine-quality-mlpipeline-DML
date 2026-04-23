# CI/CD ML Pipeline — Execution Report

**Date:** 2026-04-22
**Dataset:** UCI Wine Quality (Red) — `winequality-red.csv`
**Model:** Random Forest Classifier (100 trees)

## Pipeline Stages

| Stage | Status | Time |
|-------|--------|------|
| Data Ingestion | PASS | 1.57s |
| Preprocessing | PASS | 0.02s |
| Model Training | PASS | 0.12s |
| Evaluation | PASS | 0.04s |
| Visualization | PASS | 0.58s |

**Total runtime:** ~2.3s

## Model Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.9406 |
| ROC-AUC | 0.9554 |
| F1 Score | 0.7324 |
| Train Accuracy | 1.0000 |

Threshold: 0.75 — **PASSED**

## Confusion Matrix

|  | Predicted Low | Predicted High |
|--|---------------|----------------|
| Actual Low  | 275 | 2 |
| Actual High | 17 | 26 |

## Key Findings

- `alcohol` = top predictor (importance 0.174)
- High-quality wines: higher alcohol, lower volatile acidity, more sulphates
- Dataset imbalanced: 86.4% low quality / 13.6% high quality

## Output Files

```
outputs/
├── pipeline_report.md
└── plots/
    ├── 0_pipeline_report.png
    ├── 1_quality_distribution.png
    ├── 2_feature_correlations.png
    ├── 3_feature_distributions.png
    ├── 4_feature_importance.png
    ├── 5_confusion_matrix.png
    └── 6_roc_curve.png
```
