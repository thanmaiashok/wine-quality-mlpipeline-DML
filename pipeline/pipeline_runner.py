"""Main pipeline runner — executes all stages sequentially, stops on failure (mirrors Jenkins behavior)"""
import sys
import time

def run_stage(name, fn):
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"\n[OK] {name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[FAILED] {name} failed after {elapsed:.2f}s: {e}")
        sys.exit(1)


def main():
    print("\n" + "="*60)
    print("  CI/CD ML PIPELINE — Wine Quality Dataset")
    print("  Stages: Ingest → Preprocess → Train → Test → Visualize")
    print("="*60)

    from stage1_ingest import ingest
    from stage2_preprocess import preprocess
    from stage3_train import train
    from stage4_test import evaluate
    from stage5_visualize import visualize

    run_stage("1. Data Ingestion", ingest)
    run_stage("2. Preprocessing", preprocess)
    run_stage("3. Model Training", train)
    run_stage("4. Evaluation/Tests", evaluate)
    run_stage("5. Visualization", visualize)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE — Check outputs/plots/ for results")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
