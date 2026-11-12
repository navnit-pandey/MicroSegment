"""
MicroSegment - Production Real-Time Segment Classifier
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Trains a high-fidelity supervised classifier on discovered segments to enable
         instant, real-time segment prediction and advisory generation for new loan applicants.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

from src.advisory_engine import generate_advisory_card, ADVISORY_STRATEGY_CATALOG
from src.features import RuralFeatureEngineer


# ─────────────────────────────────────────────────────────────────────
# 1. Training & Cross-Validation
# ─────────────────────────────────────────────────────────────────────

def train_production_classifier(
    data_path: str = "data/processed/rural_micro_entrepreneurs_segmented.csv",
    model_dir: str = "models",
    fig_dir: str = "reports/figures"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 65)
    print("PHASE 7: PRODUCTION REAL-TIME CLASSIFIER")
    print("=" * 65)

    df = pd.read_csv(data_path)
    
    # Feature columns used for classification
    feature_cols = [
        "monthly_revenue", "loan_amount", "interest_concession_pct", "tenure_months",
        "repayment_ratio", "revenue_to_loan_ratio", "income_stability",
        "business_maturity", "financial_inclusion_score", "digital_adoption_score",
        "years_in_business", "num_employees", "num_delayed_installments",
        "savings_balance_ratio", "digital_payment_ratio"
    ]
    
    X = df[feature_cols].copy()
    y = df["segment_label"].values

    print(f"\n[Step 1] Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    
    # 5-fold Stratified Cross Validation
    print("\n[Step 2] Executing 5-Fold Stratified Cross-Validation (Partition Emulation Fidelity) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    y_pred_cv = cross_val_predict(rf, X, y, cv=cv)
    
    acc = accuracy_score(y, y_pred_cv)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred_cv, average="macro")
    
    print("\n  +-------------------------------------------------------------+")
    print("  |  SURROGATE PARTITION EMULATION FIDELITY (5-Fold CV)         |")
    print("  |  (Measures RF approximation of K-Means cluster partition)   |")
    print("  +-------------------------------------------------------------+")
    print(f"  |  Partition Match Acc : {acc * 100:.2f}%                                  |")
    print(f"  |  Macro Precision     : {prec:.4f}                                   |")
    print(f"  |  Macro Recall        : {rec:.4f}                                   |")
    print(f"  |  Macro F1-Score      : {f1:.4f}                                   |")
    print("  +-------------------------------------------------------------+")
    print("  [NOTE] This measures fidelity in emulating unsupervised cluster assignments,")
    print("  NOT real-world borrower credit risk prediction. Prospective field deployment")
    print("  requires validation on independent administrative datasets (NAFIS / MUDRA).")
    
    target_names = [f"Seg {i}: {ADVISORY_STRATEGY_CATALOG[i]['archetype']}" for i in range(5)]
    print("\nDetailed Per-Segment Classification Report:")
    print(classification_report(y, y_pred_cv, target_names=target_names))
    
    # Fit on all data for production deployment
    print("[Step 3] Fitting final production classifier on full portfolio ...")
    rf.fit(X, y)
    
    classifier_artifact = {
        "model": rf,
        "feature_cols": feature_cols,
        "classes": [int(c) for c in rf.classes_],
        "metrics": {
            "cv_accuracy": round(float(acc), 4),
            "cv_macro_f1": round(float(f1), 4),
            "cv_precision": round(float(prec), 4),
            "cv_recall": round(float(rec), 4)
        }
    }
    
    joblib.dump(classifier_artifact, os.path.join(model_dir, "production_classifier.joblib"))
    print("  Classifier saved -> models/production_classifier.joblib")
    
    # Confusion Matrix Visualization
    print("\n[Step 4] Plotting Confusion Matrix ...")
    cm = confusion_matrix(y, y_pred_cv)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=True,
        xticklabels=[f"Seg {i}" for i in range(5)],
        yticklabels=[f"Seg {i}" for i in range(5)],
        ax=ax
    )
    plt.title("Production Classifier: 5-Fold Stratified CV Confusion Matrix",
              fontsize=14, weight="bold", pad=15)
    plt.xlabel("Predicted Segment", fontsize=12)
    plt.ylabel("Assigned Cluster Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "classifier_confusion_matrix.png"), dpi=300)
    plt.close()
    print("  Confusion matrix saved -> reports/figures/classifier_confusion_matrix.png")
    
    print("\n" + "=" * 65)
    print("PHASE 7 COMPLETE")
    print("=" * 65)
    
    return classifier_artifact


# ─────────────────────────────────────────────────────────────────────
# 2. Production Inference Function
# ─────────────────────────────────────────────────────────────────────

class RealTimeSegmentInferenceEngine:
    """
    Production inference engine that receives raw micro-entrepreneur application
    data, runs necessary feature transformations, assigns the segment, and generates
    the operational advisory card.
    """
    def __init__(self, model_path: str = "models/production_classifier.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Run training first.")
        self.artifact = joblib.load(model_path)
        self.model = self.artifact["model"]
        self.feature_cols = self.artifact["feature_cols"]
        self.fe = RuralFeatureEngineer()
        
    def predict(self, applicant_dict: dict) -> dict:
        """
        Accepts a raw applicant dictionary and returns segment prediction,
        probabilities, confidence, and complete advisory card.
        """
        # Convert to single-row dataframe
        df_raw = pd.DataFrame([applicant_dict])
        
        # Apply feature engineering
        df_fe = self.fe.transform(df_raw)
        
        # Ensure all required features are present
        X = df_fe[self.feature_cols].copy()
        
        # Predict class & probabilities
        probs = self.model.predict_proba(X)[0]
        seg_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        prob_dict = {f"Segment_{i}": round(float(p), 4) for i, p in enumerate(probs)}
        prob_per_segment = {i: round(float(p), 4) for i, p in enumerate(probs)}
        
        # Merge engineered metrics into applicant dict for advisory generation
        enriched_data = dict(applicant_dict)
        for c in df_fe.columns:
            enriched_data[c] = df_fe[c].values[0]
                
        # Generate advisory card with fully engineered metrics
        card = generate_advisory_card(enriched_data, segment_id=seg_id)
        
        return {
            "predicted_segment_id": seg_id,
            "archetype_name": ADVISORY_STRATEGY_CATALOG[seg_id]["archetype"],
            "vernacular_title": ADVISORY_STRATEGY_CATALOG[seg_id]["vernacular_title"],
            "confidence": round(confidence, 4),
            "class_probabilities": prob_dict,
            "probabilities_by_segment_id": prob_per_segment,
            "advisory_card": card,
            "enriched_applicant": enriched_data
        }


if __name__ == "__main__":
    train_production_classifier()
