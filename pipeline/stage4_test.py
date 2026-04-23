"""Stage 4: Test & Evaluate — full classification report, fail pipeline if accuracy < threshold"""
import numpy as np
import os
import json
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
ACCURACY_THRESHOLD = 0.75


def evaluate():
    print("[STAGE 4] Running evaluation tests...")

    model = joblib.load(os.path.join(MODELS, 'rf_model.pkl'))
    X_test = np.load(os.path.join(PROC, 'X_test.npy'))
    y_test = np.load(os.path.join(PROC, 'y_test.npy'))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    results = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }
    with open(os.path.join(MODELS, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Pipeline gate: fail if below threshold
    if accuracy < ACCURACY_THRESHOLD:
        raise ValueError(
            f"[STAGE 4] PIPELINE FAILED: accuracy {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}"
        )

    print(f"\n[STAGE 4] PASSED: accuracy {accuracy:.4f} >= threshold {ACCURACY_THRESHOLD}")
    return results


if __name__ == '__main__':
    evaluate()
