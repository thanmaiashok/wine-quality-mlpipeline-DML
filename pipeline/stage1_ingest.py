"""Stage 1: Data Ingestion — fetch Wine Quality dataset from UCI ML Repo"""
import requests
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
OUTPUT = os.path.join(RAW_DIR, 'winequality-red.csv')


def ingest():
    print("[STAGE 1] Ingesting dataset from UCI ML Repository...")
    os.makedirs(RAW_DIR, exist_ok=True)
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    with open(OUTPUT, 'wb') as f:
        f.write(response.content)
    size = os.path.getsize(OUTPUT)
    print(f"[STAGE 1] Downloaded {size} bytes → {OUTPUT}")
    return OUTPUT


if __name__ == '__main__':
    ingest()
