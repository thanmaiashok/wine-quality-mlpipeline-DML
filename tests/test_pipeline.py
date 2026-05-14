"""Unit tests for pipeline stages — mirrors Jenkins test stage gate"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'etl'))

PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'winequality-red.csv')


class TestDataIngestion:
    def test_raw_file_exists(self):
        assert os.path.exists(RAW), "Raw dataset not downloaded"

    def test_raw_file_not_empty(self):
        assert os.path.getsize(RAW) > 1000, "Raw file suspiciously small"


class TestPreprocessing:
    def test_processed_files_exist(self):
        for fname in ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']:
            assert os.path.exists(os.path.join(PROC, fname)), f"Missing {fname}"

    def test_train_test_shapes_match(self):
        X_train = np.load(os.path.join(PROC, 'X_train.npy'))
        y_train = np.load(os.path.join(PROC, 'y_train.npy'))
        X_test = np.load(os.path.join(PROC, 'X_test.npy'))
        y_test = np.load(os.path.join(PROC, 'y_test.npy'))
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

    def test_feature_count(self):
        X_train = np.load(os.path.join(PROC, 'X_train.npy'))
        assert X_train.shape[1] == 11, "Expected 11 wine features"

    def test_labels_binary(self):
        y_train = np.load(os.path.join(PROC, 'y_train.npy'))
        y_test = np.load(os.path.join(PROC, 'y_test.npy'))
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})


class TestModel:
    def test_model_file_exists(self):
        assert os.path.exists(os.path.join(MODELS, 'rf_model.pkl')), "Model not saved"

    def test_model_accuracy_threshold(self):
        import json
        results_path = os.path.join(MODELS, 'eval_results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            assert results['accuracy'] >= 0.75, f"Accuracy {results['accuracy']:.4f} below threshold"

    def test_model_predicts(self):
        import joblib
        model = joblib.load(os.path.join(MODELS, 'rf_model.pkl'))
        X_test = np.load(os.path.join(PROC, 'X_test.npy'))
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})
