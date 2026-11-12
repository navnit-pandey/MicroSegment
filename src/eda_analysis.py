"""
MicroSegment - Exploratory Data Analysis & Visualization Script
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Generate distributions, skew metrics, correlation heatmaps,
         and save figures to reports/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda_and_generate_plots(
    raw_path: str = "data/raw/rural_micro_entrepreneurs_synthetic.csv",
    eng_path: str = "data/processed/rural_micro_entrepreneurs_engineered.csv",
    fig_dir: str = "reports/figures"
):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    df_raw = pd.read_csv(raw_path)
    df_eng = pd.read_csv(eng_path)
    
    # -------------------------------------------------------------
    # 1. Skewness and Transformation Analysis Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Financial Feature Distributions & Skewness Analysis (N=4,000)", fontsize=16, weight="bold")
    
    # Monthly Revenue
    sns.histplot(df_raw["monthly_revenue"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=35)
    axes[0, 0].set_title(f"Monthly Revenue (₹) [Skew: {df_raw['monthly_revenue'].skew():.2f}]", fontsize=12, weight="semibold")
    axes[0, 0].set_xlabel("Monthly Revenue (INR)")
    
    # Loan Amount
    sns.histplot(df_raw["loan_amount"], kde=True, ax=axes[0, 1], color="#2b8f65", bins=35)
    axes[0, 1].set_title(f"Loan Amount (₹) [Skew: {df_raw['loan_amount'].skew():.2f}]", fontsize=12, weight="semibold")
    axes[0, 1].set_xlabel("Loan Amount (INR)")
    
    # Repayment Ratio
    sns.histplot(df_eng["repayment_ratio"], kde=True, ax=axes[1, 0], color="#8f2b2b", bins=35)
    axes[1, 0].set_title(f"Engineered Repayment Ratio [Skew: {df_eng['repayment_ratio'].skew():.2f}]", fontsize=12, weight="semibold")
    axes[1, 0].set_xlabel("Repayment Integrity Score (0 to 1)")
    
    # Revenue to Loan Ratio
    sns.histplot(df_eng["revenue_to_loan_ratio"], kde=True, ax=axes[1, 1], color="#852b8f", bins=35)
    axes[1, 1].set_title(f"Annual Revenue / Loan Ratio [Skew: {df_eng['revenue_to_loan_ratio'].skew():.2f}]", fontsize=12, weight="semibold")
    axes[1, 1].set_xlabel("Revenue to Loan Ratio")
    
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "eda_distributions.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Correlation Matrix Heatmap
    # -------------------------------------------------------------
    corr_cols = [
        "monthly_revenue", "loan_amount", "years_in_business", "num_employees",
        "repayment_ratio", "revenue_to_loan_ratio", "income_stability",
        "business_maturity", "financial_inclusion_score", "digital_adoption_score"
    ]
    corr_mat = df_eng[corr_cols].corr()
    
    plt.figure(figsize=(11, 8.5))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.8, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f",
                annot_kws={"size": 9})
    plt.title("Correlation Matrix of Financial, Operational & Behavioral Indices", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "correlation_heatmap.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Revenue by Business Sector & Location Type
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    order = df_raw.groupby("business_sector")["monthly_revenue"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df_raw,
        x="business_sector",
        y="monthly_revenue",
        hue="location_type",
        order=order,
        palette="Set2",
        showfliers=False
    )
    plt.title("Monthly Revenue (₹) by Business Sector and Location Tier", fontsize=14, weight="bold")
    plt.xlabel("Business Sector", fontsize=11, weight="semibold")
    plt.ylabel("Monthly Revenue (INR)", fontsize=11, weight="semibold")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Location Type", frameon=True)
    plt.tight_layout()
    p3 = os.path.join(fig_dir, "sector_revenue_boxplot.png")
    plt.savefig(p3, dpi=300)
    plt.close()

    # Compute exact numerical summary statistics
    skew_table = pd.DataFrame({
        "Feature": ["monthly_revenue", "loan_amount", "revenue_to_loan_ratio", "years_in_business"],
        "Raw_Skewness": [
            df_raw["monthly_revenue"].skew(),
            df_raw["loan_amount"].skew(),
            df_eng["revenue_to_loan_ratio"].skew(),
            df_raw["years_in_business"].skew()
        ]
    })
    
    return skew_table, corr_mat

if __name__ == "__main__":
    skew_df, corr = run_eda_and_generate_plots()
    print("EDA completed successfully!")
    print("\nFeature Skewness Metrics:")
    print(skew_df.round(3).to_string(index=False))
