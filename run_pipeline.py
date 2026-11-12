"""
MicroSegment - Master Pipeline Runner
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Orchestrates the entire data science and machine learning workflow end-to-end
         with standardized execution, timing, and error handling.
"""

import os
import sys
import time
import argparse
import subprocess

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_banner(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f"  STEP {step_num}: {title.upper()}")
    print("=" * 70)


def run_pipeline(skip_data_gen: bool = False, run_app: bool = False):
    total_start = time.time()
    print("*" * 70)
    print("  MICROSEGMENT: RURAL MICRO-ENTREPRENEUR SEGMENTATION PIPELINE")
    print("  End-to-End Orchestrator")
    print("*" * 70)

    # Step 1: Data Generation
    if not skip_data_gen:
        print_banner(1, "Synthetic Data Generation (N=4,000)")
        from src.data_generator import generate_synthetic_microfinance_data
        df_raw, meta = generate_synthetic_microfinance_data(n_samples=4000, random_state=42)
        os.makedirs("data/raw", exist_ok=True)
        df_raw.to_csv("data/raw/rural_micro_entrepreneurs_synthetic.csv", index=False)
        print(f"  [SUCCESS] Generated dataset shape: {df_raw.shape} saved -> data/raw/rural_micro_entrepreneurs_synthetic.csv")
    else:
        print("\n  [INFO] Skipping Step 1 (Data Generation) as requested.")

    # Step 2: Feature Engineering & Preprocessing Pipeline
    print_banner(2, "Feature Engineering & Scikit-Learn Pipeline Fitting")
    from src.features import process_and_save_dataset
    df_eng, X_trans, feat_names = process_and_save_dataset()
    print(f"  [SUCCESS] Engineered feature matrix shape: {X_trans.shape} with {len(feat_names)} features.")

    # Step 3: Exploratory Data Analysis & Diagnostic Visualizations
    print_banner(3, "Exploratory Data Analysis & Diagnostic Visualizations")
    from src.eda_analysis import run_eda_and_generate_plots
    run_eda_and_generate_plots()
    print("  [SUCCESS] EDA visual artifacts created.")

    # Step 4: Segmentation Modelling & Multi-Metric Validation
    print_banner(4, "Clustering Modelling, Metric Sweeps & Stability Analysis")
    from src.clustering import run_segmentation
    model, labels, metrics, stability = run_segmentation(final_k=5)
    print(f"  [SUCCESS] Clustered {len(labels)} entrepreneurs into k=5 optimal segments.")

    # Step 5: Cluster Profiling & Persona Generation
    print_banner(5, "Statistical Profiling & Supervised Surrogate Explainability")
    from src.profiling import run_profiling
    profiles, summary_df, surrogate = run_profiling()
    print("  [SUCCESS] Persona profiles and surrogate explainer saved.")

    # Step 6: Advisory Recommendation Engine
    print_banner(6, "Operational Advisory Strategy Assignment")
    from src.advisory_engine import batch_assign_advisory_strategies
    import pandas as pd
    df_seg = pd.read_csv("data/processed/rural_micro_entrepreneurs_segmented.csv")
    df_adv = batch_assign_advisory_strategies(df_seg)
    df_adv.to_csv("data/processed/rural_micro_entrepreneurs_with_advisory.csv", index=False)
    print(f"  [SUCCESS] Enriched dataset saved with advisory strategies.")

    # Step 7: Production Real-Time Classifier
    print_banner(7, "Training Real-Time Production Classifier (5-Fold CV)")
    from src.classifier import train_production_classifier
    clf_artifact = train_production_classifier()
    print(f"  [SUCCESS] Classifier trained. CV Macro F1: {clf_artifact['metrics']['cv_macro_f1'] * 100:.2f}%")

    elapsed = time.time() - total_start
    print("\n" + "*" * 70)
    print(f"  ALL 7 PHASES COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("*" * 70)

    # Optional Launch Streamlit
    if run_app:
        print("\n  Launching Streamlit Interactive Dashboard ...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app/main.py"]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroSegment End-to-End Pipeline Runner")
    parser.add_argument("--skip-data", action="store_true", help="Skip synthetic data generation if raw data exists")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application after pipeline execution")
    args = parser.parse_args()

    run_pipeline(skip_data_gen=args.skip_data, run_app=args.app)
