"""Stage 2: Preprocessing — clean, encode, split Wine Quality data"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'winequality-red.csv')
PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def preprocess():
    print("[STAGE 2] Preprocessing data...")
    os.makedirs(PROC, exist_ok=True)

    df = pd.read_csv(RAW, sep=';')
    print(f"[STAGE 2] Shape: {df.shape}, nulls: {df.isnull().sum().sum()}")

    # Binary classification: quality >= 7 = high quality
    df['label'] = (df['quality'] >= 7).astype(int)
    X = df.drop(columns=['quality', 'label'])
    y = df['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save splits
    np.save(os.path.join(PROC, 'X_train.npy'), X_train)
    np.save(os.path.join(PROC, 'X_test.npy'), X_test)
    np.save(os.path.join(PROC, 'y_train.npy'), y_train)
    np.save(os.path.join(PROC, 'y_test.npy'), y_test)

    # Save feature names and raw df for viz
    df.to_csv(os.path.join(PROC, 'wine_full.csv'), index=False)

    meta = {
        'features': list(X.columns),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'class_balance': y.value_counts().to_dict()
    }
    with open(os.path.join(PROC, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[STAGE 2] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[STAGE 2] Class balance: {meta['class_balance']}")
    return PROC


if __name__ == '__main__':
    preprocess()
