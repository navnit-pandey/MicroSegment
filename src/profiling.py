"""
MicroSegment - Cluster Profiling & Persona Generation Module
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Deep statistical profiling, radar chart comparisons, demographic composition,
         and supervised surrogate tree interpretability for the 5 rural entrepreneur segments.
"""

import os
import sys
import json
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score


# ─────────────────────────────────────────────────────────────────────
# 1. Archetype Persona Definitions & Metadata
# ─────────────────────────────────────────────────────────────────────

SEGMENT_ARCHETYPES = {
    0: {
        "name": "Agri-Allied Commercial Scalers",
        "vernacular_title": "Vikas Pragati (Commercial Producers)",
        "tagline": "High-capacity commercial units ready for modernization and capital deepening.",
        "risk_tier": "Low to Moderate",
        "digital_tier": "Intermediate",
        "primary_sectors": ["Agri-Allied & Dairy", "Food Processing", "Horticulture"],
        "advisory_focus": "Equipment leasing, cold chain/logistics, B2B wholesale forward-contracts, GST migration.",
        "icon": "🚜"
    },
    1: {
        "name": "Digital-First Youth Services",
        "vernacular_title": "Digital Yuva (Tech-Enabled Services)",
        "tagline": "Mobile-first younger entrepreneurs running agile tech/service hubs.",
        "risk_tier": "Moderate (High Growth, Short Track Record)",
        "digital_tier": "Advanced",
        "primary_sectors": ["Electronics & Repair", "Digital Services / CSC", "Modern Retail"],
        "advisory_focus": "Working capital limits, digital marketing, ONDC/e-commerce onboarding, inventory software.",
        "icon": "📱"
    },
    2: {
        "name": "Vulnerable Seasonal Producers",
        "vernacular_title": "Sankat-Prone (Vulnerable Micro-Farmers)",
        "tagline": "Margin-thin, weather-dependent micro-enterprises facing severe cashflow volatility.",
        "risk_tier": "High / Critical",
        "digital_tier": "Nascent / Basic",
        "primary_sectors": ["Crops & Cultivation", "Seasonal Produce", "Basic Trading"],
        "advisory_focus": "Restructured seasonal balloon EMIs, crop micro-insurance, SHG solidarity, emergency credit.",
        "icon": "🌾"
    },
    3: {
        "name": "Women SHG Collective Artisans",
        "vernacular_title": "Mahila Shakti (Collective Artisans)",
        "tagline": "Impeccable repayment discipline with constrained capital and strong social capital.",
        "risk_tier": "Very Low (Pristine Repayment Track)",
        "digital_tier": "Emerging / Basic",
        "primary_sectors": ["Handloom & Textiles", "Handicrafts", "Small-scale Processing"],
        "advisory_focus": "Direct artisan aggregation, exhibition/fair linkages, collective bulk procurement, digital QR literacy.",
        "icon": "🧵"
    },
    4: {
        "name": "Stable Subsistence Retailers",
        "vernacular_title": "Gramin Kirana (Village Retail & Trading)",
        "tagline": "Resilient, long-established neighborhood stores with steady predictable cashflows.",
        "risk_tier": "Low (Mature & Resilient)",
        "digital_tier": "Low to Intermediate",
        "primary_sectors": ["Kirana & Grocery", "Local General Store", "Traditional Retail"],
        "advisory_focus": "Supplier credit formalization, FMCG margin optimization, UPI soundbox adoption, bookkeeping.",
        "icon": "🏪"
    }
}


# ─────────────────────────────────────────────────────────────────────
# 2. Comprehensive Statistical Profiling
# ─────────────────────────────────────────────────────────────────────

def compute_cluster_profiles(df: pd.DataFrame) -> dict:
    """
    Computes rigorous univariate and bivariate descriptive statistics
    per cluster across financial, operational, behavioral, and demographic features.
    """
    numeric_cols = [
        "monthly_revenue", "loan_amount", "interest_concession_pct", "tenure_months",
        "repayment_ratio", "revenue_to_loan_ratio", "income_stability",
        "business_maturity", "financial_inclusion_score", "digital_adoption_score",
        "years_in_business", "num_employees", "num_delayed_installments",
        "digital_payment_ratio", "savings_balance_ratio"
    ]
    
    profiles = {}
    total_records = len(df)
    
    for seg_id, archetype in SEGMENT_ARCHETYPES.items():
        sub = df[df["segment_label"] == seg_id]
        seg_size = len(sub)
        seg_pct = seg_size / total_records * 100
        
        stat_dict = {
            "segment_id": seg_id,
            "archetype_name": archetype["name"],
            "vernacular_title": archetype["vernacular_title"],
            "tagline": archetype["tagline"],
            "risk_tier": archetype["risk_tier"],
            "digital_tier": archetype["digital_tier"],
            "advisory_focus": archetype["advisory_focus"],
            "icon": archetype["icon"],
            "size": int(seg_size),
            "percentage_of_portfolio": round(seg_pct, 2),
            "metrics": {},
            "demographics": {}
        }
        
        for col in numeric_cols:
            if col in sub.columns:
                stat_dict["metrics"][col] = {
                    "mean": round(float(sub[col].mean()), 2),
                    "std": round(float(sub[col].std()), 2),
                    "median": round(float(sub[col].median()), 2),
                    "p25": round(float(sub[col].quantile(0.25)), 2),
                    "p75": round(float(sub[col].quantile(0.75)), 2),
                    "min": round(float(sub[col].min()), 2),
                    "max": round(float(sub[col].max()), 2)
                }
                
        # Categorical distributions
        for cat_col in ["business_sector", "gender", "market_access_type", "mobile_banking_usage", "shg_cooperative_member", "education", "region"]:
            if cat_col in sub.columns:
                top_cats = sub[cat_col].value_counts(normalize=True).head(5).to_dict()
                stat_dict["demographics"][cat_col] = {str(k): round(float(v * 100), 1) for k, v in top_cats.items()}
                
        profiles[str(seg_id)] = stat_dict
        
    return profiles


# ─────────────────────────────────────────────────────────────────────
# 3. Visualization: Radar / Spider Chart
# ─────────────────────────────────────────────────────────────────────

def plot_radar_profiles(df: pd.DataFrame, fig_path: str = "reports/figures/segment_radar_profiles.png"):
    """
    Renders a multi-segment radar chart comparing the 5 segments
    on 7 normalized behavioral and financial dimensions.
    """
    # Define normalized indicators (0 to 1 scale)
    radar_features = [
        ("Revenue Scale", "monthly_revenue", 100000.0),
        ("Credit Capacity", "loan_amount", 250000.0),
        ("Repayment Integrity", "repayment_ratio", 1.0),
        ("Cashflow Stability", "income_stability", 1.0),
        ("Digital Maturity", "digital_adoption_score", 1.0),
        ("Inclusion & SHG", "financial_inclusion_score", 1.0),
        ("Business Maturity", "business_maturity", 1.0)
    ]
    
    categories = [rf[0] for rf in radar_features]
    num_vars = len(categories)
    
    # Compute angles for radar
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.title("MicroSegment: Comparative Persona Radar Profiles\n(7 Core Financial & Behavioral Dimensions)",
              size=15, weight="bold", pad=30, color="#1e293b")
    
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#db2777", "#059669"]
    
    for seg_id, archetype in SEGMENT_ARCHETYPES.items():
        sub = df[df["segment_label"] == seg_id]
        values = []
        for _, col, max_val in radar_features:
            val = sub[col].mean() / max_val
            values.append(min(max(val, 0.0), 1.0))
        values += values[:1]  # complete loop
        
        ax.plot(angles, values, linewidth=2.5, linestyle="solid",
                label=f"Seg {seg_id}: {archetype['name']}", color=colors[seg_id])
        ax.fill(angles, values, color=colors[seg_id], alpha=0.12)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight="bold", color="#334155")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9, color="#64748b")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10.5, frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Radar profile saved -> {fig_path}")


# ─────────────────────────────────────────────────────────────────────
# 4. Visualization: Segment Distributions & Demographic Composition
# ─────────────────────────────────────────────────────────────────────

def plot_segment_distributions(df: pd.DataFrame, fig_dir: str = "reports/figures"):
    """
    Renders box plots and compositional demographic bar charts across clusters.
    """
    palette = ["#2563eb", "#7c3aed", "#dc2626", "#db2777", "#059669"]
    
    # 1. Financial & Behavioral Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Key Metric Distributions Across Segments", fontsize=18, weight="bold", y=0.98)
    
    plot_specs = [
        ("monthly_revenue", "Monthly Revenue (INR)", axes[0, 0]),
        ("repayment_ratio", "Repayment Integrity Ratio (0-1)", axes[0, 1]),
        ("digital_adoption_score", "Digital Adoption Score (0-1)", axes[1, 0]),
        ("income_stability", "Income Stability Score (0-1)", axes[1, 1])
    ]
    
    for col, title, ax in plot_specs:
        sns.boxplot(data=df, x="segment_label", y=col, ax=ax, hue="segment_label", palette=palette, legend=False, fliersize=2)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Segment", fontsize=11)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"Seg {i}" for i in range(5)])
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "segment_boxplots.png"), dpi=300)
    plt.close()
    
    # 2. Demographic Composition Breakdown
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Demographic & Structural Composition by Segment", fontsize=16, weight="bold", y=1.02)
    
    # Gender
    gender_ct = pd.crosstab(df["segment_label"], df["gender"], normalize="index") * 100
    gender_ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Gender Distribution (%)", fontsize=12, weight="bold")
    axes[0].set_xlabel("Segment")
    axes[0].set_xticklabels([f"Seg {i}" for i in range(5)], rotation=0)
    axes[0].legend(loc="upper right")
    
    # Primary Sales Channel
    sales_ct = pd.crosstab(df["segment_label"], df["market_access_type"], normalize="index") * 100
    sales_ct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Market Access Channels (%)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Segment")
    axes[1].set_xticklabels([f"Seg {i}" for i in range(5)], rotation=0)
    axes[1].legend(loc="upper right")
    
    # SHG Membership
    shg_ct = pd.crosstab(df["segment_label"], df["shg_cooperative_member"], normalize="index") * 100
    shg_ct.plot(kind="bar", stacked=True, ax=axes[2], color=["#94a3b8", "#ec4899"], edgecolor="black", linewidth=0.5)
    axes[2].set_title("SHG Membership (%)", fontsize=12, weight="bold")
    axes[2].set_xlabel("Segment")
    axes[2].set_xticklabels([f"Seg {i}" for i in range(5)], rotation=0)
    axes[2].legend(["Non-SHG", "SHG Member"], loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "segment_demographics.png"), dpi=300)
    plt.close()
    print("  Distributions & Demographics saved -> reports/figures/segment_boxplots.png, segment_demographics.png")


# ─────────────────────────────────────────────────────────────────────
# 5. Supervised Surrogate Interpretability (Explainability)
# ─────────────────────────────────────────────────────────────────────

def fit_surrogate_explainer(df: pd.DataFrame, model_dir: str = "models", fig_dir: str = "reports/figures"):
    """
    Fits a Random Forest surrogate model to explain the unsupervised cluster
    assignments using original engineered features. Computes feature importance
    and cross-validated fidelity.
    """
    feature_cols = [
        "monthly_revenue", "loan_amount", "interest_concession_pct", "tenure_months",
        "repayment_ratio", "revenue_to_loan_ratio", "income_stability",
        "business_maturity", "financial_inclusion_score", "digital_adoption_score",
        "years_in_business", "num_employees", "num_delayed_installments"
    ]
    
    X = df[feature_cols].copy()
    y = df["segment_label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    surrogate = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )
    surrogate.fit(X_train, y_train)
    
    y_pred = surrogate.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n[Surrogate Model Fidelity] Test Accuracy = {acc:.4f}, Macro F1 = {f1:.4f}")
    
    # Save surrogate model
    joblib.dump(surrogate, os.path.join(model_dir, "surrogate_rf.joblib"))
    
    # Feature Importance Plot
    importances = surrogate.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    y_labels = [feature_cols[i] for i in sorted_idx]
    sns.barplot(x=importances[sorted_idx], y=y_labels, hue=y_labels, palette="Blues_r", legend=False)
    plt.xlabel("Relative Importance", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "surrogate_feature_importance.png"), dpi=300)
    plt.close()
    
    print(f"  Surrogate model saved -> models/surrogate_rf.joblib")
    print(f"  Surrogate feature importance saved -> reports/figures/surrogate_feature_importance.png")
    
    return surrogate, {col: round(float(imp), 4) for col, imp in zip(feature_cols, importances)}


# ─────────────────────────────────────────────────────────────────────
# 6. Master Runner
# ─────────────────────────────────────────────────────────────────────

def run_profiling():
    print("=" * 65)
    print("PHASE 5: CLUSTER PROFILING & PERSONA GENERATION")
    print("=" * 65)
    
    # Load segmented data
    segmented_path = "data/processed/rural_micro_entrepreneurs_segmented.csv"
    if not os.path.exists(segmented_path):
        raise FileNotFoundError(f"Missing {segmented_path}. Run Phase 4 first.")
        
    df = pd.read_csv(segmented_path)
    
    # Compute full statistical profiles
    print("\n[Step 1] Computing statistical profiles across all 5 segments ...")
    profiles = compute_cluster_profiles(df)
    
    # Save JSON profiles
    with open("models/segment_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print("  Segment profiles saved -> models/segment_profiles.json")
    
    # Flatten to summary CSV for reports
    summary_rows = []
    for seg_id, p in profiles.items():
        row = {
            "Segment ID": seg_id,
            "Archetype": p["archetype_name"],
            "Portfolio Pct": f"{p['percentage_of_portfolio']}%",
            "Mean Revenue (INR)": f"INR {p['metrics']['monthly_revenue']['mean']:,.0f}",
            "Mean Loan (INR)": f"INR {p['metrics']['loan_amount']['mean']:,.0f}",
            "Repayment Ratio": p["metrics"]["repayment_ratio"]["mean"],
            "Income Stability": p["metrics"]["income_stability"]["mean"],
            "Digital Score": p["metrics"]["digital_adoption_score"]["mean"],
            "Inclusion Score": p["metrics"]["financial_inclusion_score"]["mean"],
            "Years in Business": p["metrics"]["years_in_business"]["mean"],
            "Employees": p["metrics"]["num_employees"]["mean"],
            "Delinquencies": p["metrics"]["num_delayed_installments"]["mean"],
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/segment_statistical_summary.csv", index=False)
    print("  Statistical summary table saved -> reports/segment_statistical_summary.csv")
    
    print("\n  Summary Table:")
    print(summary_df.to_string(index=False))
    
    # Visualizations
    print("\n[Step 2] Rendering comparative radar profiles ...")
    plot_radar_profiles(df)
    
    print("\n[Step 3] Rendering distributions and demographic breakdowns ...")
    plot_segment_distributions(df)
    
    print("\n[Step 4] Fitting surrogate interpretability model ...")
    surrogate, feat_imp = fit_surrogate_explainer(df)
    
    print("\n" + "=" * 65)
    print("PHASE 5 COMPLETE")
    print("=" * 65)
    return profiles, summary_df, surrogate


if __name__ == "__main__":
    run_profiling()
