"""
MicroSegment - Feature Engineering & Preprocessing Pipeline
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Domain-specific feature engineering, skew transformation (Yeo-Johnson/Log),
         imputation, scaling, and end-to-end scikit-learn preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import joblib
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler

class RuralFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to compute domain-specific rural microfinance features:
    1. repayment_ratio (repayment integrity accounting for delays and default penalty)
    2. revenue_to_loan_ratio (annualized revenue-to-loan leverage ratio)
    3. business_maturity (composite enterprise vintage and workforce scale)
    4. income_stability (inverse of income variability/seasonality)
    5. financial_inclusion_score (composite savings, subvention, and SHG integration)
    6. digital_adoption_score (composite mobile channel and UPI usage)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Repayment Ratio (Penalizing delayed installments and default history)
        tenure = df["tenure_months"].replace(0, 12).fillna(12)
        penalty = (df["num_delayed_installments"].fillna(0) / tenure) * 0.5 + (df["has_defaulted_past"].fillna(0) * 0.25)
        df["repayment_ratio"] = (df["repayment_on_time_ratio"].fillna(0.8) - penalty).clip(0.0, 1.0)
        
        # 2. Revenue to Loan Ratio (Annualized Revenue Coverage)
        df["revenue_to_loan_ratio"] = ((df["monthly_revenue"] * 12.0) / df["loan_amount"].replace(0, np.nan)).clip(0.1, 20.0)
        
        # 3. Income Stability (1 - income variability)
        df["income_stability"] = (1.0 - df["income_variability"].fillna(0.35)).clip(0.0, 1.0)
        
        # 4. Business Maturity Index (0 to 1 scale)
        norm_years = (df["years_in_business"].fillna(5.0) / 20.0).clip(0.0, 1.0)
        norm_emp = (df["num_employees"].fillna(0) / 4.0).clip(0.0, 1.0)
        norm_age = ((df["age"].fillna(40) - 18) / 50.0).clip(0.0, 1.0)
        df["business_maturity"] = (0.50 * norm_years + 0.30 * norm_emp + 0.20 * norm_age).round(3)
        
        # 5. Financial Inclusion Score (0 to 1 scale)
        savings_freq_map = {"Never": 0.0, "Irregular": 0.25, "Monthly": 0.70, "Weekly": 1.0}
        freq_score = df["savings_frequency"].map(savings_freq_map).fillna(0.4)
        bal_ratio = (df["savings_balance_ratio"].fillna(1.0) / 3.0).clip(0.0, 1.0)
        concession_score = (df["interest_concession_pct"].fillna(2.5) / 5.0).clip(0.0, 1.0)
        shg_score = df["shg_cooperative_member"].fillna(0).astype(float)
        df["financial_inclusion_score"] = (0.35 * freq_score + 0.25 * bal_ratio + 0.25 * shg_score + 0.15 * concession_score).round(3)
        
        # 6. Digital Adoption Score (0 to 1 scale)
        mobile_map = {"None": 0.0, "SMS/USSD": 0.40, "Smartphone App/UPI": 1.0}
        mobile_score = df["mobile_banking_usage"].map(mobile_map).fillna(0.40)
        digital_vol = df["digital_payment_ratio"].fillna(0.20).clip(0.0, 1.0)
        df["digital_adoption_score"] = (0.45 * mobile_score + 0.55 * digital_vol).round(3)
        
        return df

def build_preprocessing_pipeline() -> Tuple[RuralFeatureEngineer, ColumnTransformer, List[str]]:
    """
    Constructs the end-to-end scikit-learn preprocessing pipeline:
    - Skewed continuous features: Median Imputation -> Yeo-Johnson PowerTransform -> StandardScaler
    - Regular continuous & bounded engineered features: Median Imputation -> StandardScaler
    - Categorical features: Most-frequent Imputation -> OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    skewed_features = [
        "loan_amount",
        "monthly_revenue",
        "revenue_to_loan_ratio",
        "years_in_business"
    ]
    
    regular_numeric_features = [
        "age",
        "interest_concession_pct",
        "income_stability",
        "repayment_ratio",
        "business_maturity",
        "financial_inclusion_score",
        "digital_adoption_score",
        "num_employees",
        "savings_balance_ratio"
    ]
    
    categorical_features = [
        # Socio-Demographic & Operational Context Features:
        # 1. Gender: Included specifically to support affirmative policy mandates under NRLM /
        #    SHG-Bank Linkage (recognizing women collective enterprises for interest subvention).
        # 2. Education: Operational proxy for tailoring financial literacy & extension curricula.
        # 3. Location Type & Region: Captures agro-climatic seasonality, crop cycles, and market density.
        # 4. Sector & Market Access: Captures supply chain and value addition characteristics.
        # NOTE: Caste, religion, and marital status are STRICTLY EXCLUDED to prevent unlawful bias.
        "gender",
        "education",
        "location_type",
        "region",
        "business_sector",
        "market_access_type"
    ]
    
    skewed_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True))
    ])
    
    regular_numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("skew", skewed_transformer, skewed_features),
            ("num", regular_numeric_transformer, regular_numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    feature_engineer = RuralFeatureEngineer()
    return feature_engineer, preprocessor, (skewed_features + regular_numeric_features + categorical_features)

def process_and_save_dataset(
    raw_csv_path: str = "data/raw/rural_micro_entrepreneurs_synthetic.csv",
    out_dir: str = "data/processed",
    model_dir: str = "models"
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Executes feature engineering and preprocessing pipeline, saves processed artifacts.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    df_raw = pd.read_csv(raw_csv_path)
    
    # 1. Feature Engineering
    engineer, preprocessor, input_cols = build_preprocessing_pipeline()
    df_engineered = engineer.transform(df_raw)
    
    # Save unscaled engineered dataset for interpretability and profiling
    df_engineered.to_csv(os.path.join(out_dir, "rural_micro_entrepreneurs_engineered.csv"), index=False)
    
    # 2. Fit and Transform Preprocessing Pipeline
    X_processed = preprocessor.fit_transform(df_engineered)
    
    # Extract feature names
    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
        ["gender", "education", "location_type", "region", "business_sector", "market_access_type"]
    )
    all_feature_names = (
        ["loan_amount_yj", "monthly_revenue_yj", "revenue_to_loan_ratio_yj", "years_in_business_yj"] +
        [
            "age", "interest_concession_pct", "income_stability", "repayment_ratio",
            "business_maturity", "financial_inclusion_score", "digital_adoption_score",
            "num_employees", "savings_balance_ratio"
        ] +
        list(cat_names)
    )
    
    # Save transformed matrix as dataframe
    df_processed = pd.DataFrame(X_processed, columns=all_feature_names)
    df_processed["entrepreneur_id"] = df_raw["entrepreneur_id"]
    df_processed.to_csv(os.path.join(out_dir, "rural_micro_entrepreneurs_scaled_features.csv"), index=False)
    
    # Save pipeline object
    joblib.dump({"engineer": engineer, "preprocessor": preprocessor, "feature_names": all_feature_names}, 
                os.path.join(model_dir, "preprocessor_pipeline.joblib"))
    
    return df_engineered, X_processed, all_feature_names

if __name__ == "__main__":
    df_eng, X_trans, feat_names = process_and_save_dataset()
    print("Feature engineering and pipeline transformation complete!")
    print(f"Engineered dataset shape: {df_eng.shape}")
    print(f"Scaled feature matrix shape: {X_trans.shape}")
    print(f"Total features after OneHotEncoding: {len(feat_names)}")
    print("\nEngineered Feature Summary (Mean / Std):")
    cols_to_show = ["repayment_ratio", "revenue_to_loan_ratio", "business_maturity", 
                    "income_stability", "financial_inclusion_score", "digital_adoption_score"]
    print(df_eng[cols_to_show].describe().round(3).T[["mean", "std", "min", "50%", "max"]].to_string())
