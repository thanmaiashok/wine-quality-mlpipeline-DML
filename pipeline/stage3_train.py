"""Stage 3: Train — Random Forest classifier on Wine Quality"""
import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')


def train():
    print("[STAGE 3] Training Random Forest model...")
    os.makedirs(MODELS, exist_ok=True)

    X_train = np.load(os.path.join(PROC, 'X_train.npy'))
    y_train = np.load(os.path.join(PROC, 'y_train.npy'))
    X_test = np.load(os.path.join(PROC, 'X_test.npy'))
    y_test = np.load(os.path.join(PROC, 'y_test.npy'))

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"[STAGE 3] Train accuracy: {train_acc:.4f}")
    print(f"[STAGE 3] Test accuracy:  {test_acc:.4f}")

    model_path = os.path.join(MODELS, 'rf_model.pkl')
    joblib.dump(model, model_path)

    metrics = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    with open(os.path.join(MODELS, 'train_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[STAGE 3] Model saved → {model_path}")
    return model_path


if __name__ == '__main__':
    train()
