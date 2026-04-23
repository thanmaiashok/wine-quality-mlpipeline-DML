"""Stage 5: Visualize — generate 6 plots showing data insights + model performance"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')


def plot_quality_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Wine Quality Dataset — Distribution Analysis', fontsize=14, fontweight='bold')

    sns.countplot(x='quality', data=df, ax=axes[0], hue='quality', palette='viridis', legend=False)
    axes[0].set_title('Wine Quality Score Distribution')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Count')

    label_counts = df['label'].value_counts()
    axes[1].pie(label_counts, labels=['Low Quality (< 7)', 'High Quality (>= 7)'],
                autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90)
    axes[1].set_title('Binary Classification Balance')

    plt.tight_layout()
    path = os.path.join(PLOTS, '1_quality_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STAGE 5] Saved: {path}")


def plot_feature_correlations(df):
    fig, ax = plt.subplots(figsize=(12, 9))
    numeric = df.drop(columns=['label'])
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap — Wine Quality Dataset', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS, '2_feature_correlations.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STAGE 5] Saved: {path}")


def plot_feature_distributions(df):
    features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'pH', 'density']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Key Feature Distributions by Wine Quality', fontsize=13, fontweight='bold')
    axes = axes.flatten()
    for i, feat in enumerate(features):
        sns.boxplot(x='label', y=feat, data=df, ax=axes[i],
                    hue='label', palette={0: '#FF6B6B', 1: '#4ECDC4'}, legend=False)
        axes[i].set_title(feat.title())
        axes[i].set_xlabel('Quality (0=Low, 1=High)')
        axes[i].set_ylabel(feat)
    plt.tight_layout()
    path = os.path.join(PLOTS, '3_feature_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STAGE 5] Saved: {path}")


def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(features)), importances[indices], color=sns.color_palette('viridis', len(features)))
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Random Forest — Feature Importance', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance Score')
    for bar, imp in zip(bars, importances[indices]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{imp:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS, '4_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STAGE 5] Saved: {path}")


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low Quality', 'High Quality'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Random Forest Classifier', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS, '5_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STAGE 5] Saved: {path}")


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='#4ECDC4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#4ECDC4')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Wine Quality Classifier', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(PLOTS, '6_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STAGE 5] Saved: {path}")


def visualize():
    print("[STAGE 5] Generating visualizations...")
    df = pd.read_csv(os.path.join(PROC, 'wine_full.csv'))
    with open(os.path.join(PROC, 'meta.json')) as f:
        meta = json.load(f)
    model = joblib.load(os.path.join(MODELS, 'rf_model.pkl'))
    X_test = np.load(os.path.join(PROC, 'X_test.npy'))
    y_test = np.load(os.path.join(PROC, 'y_test.npy'))

    plot_quality_distribution(df)
    plot_feature_correlations(df)
    plot_feature_distributions(df)
    plot_feature_importance(model, meta['features'])
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)

    print(f"\n[STAGE 5] All 6 plots saved to {PLOTS}")


if __name__ == '__main__':
    visualize()
