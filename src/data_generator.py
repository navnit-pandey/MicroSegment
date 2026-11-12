"""
MicroSegment - Rural Micro-Entrepreneur Synthetic Dataset Generator
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Generate realistic, reproducible synthetic data for rural micro-entrepreneurs
         receiving concessional credit in India.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

def generate_synthetic_microfinance_data(
    n_samples: int = 4000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Generates a realistic synthetic dataset representing rural micro-entrepreneurs
    with 5 latent behavioral/structural archetypes, realistic correlations,
    heteroskedastic noise, empirical missingness, and heavy-tailed outliers.
    """
    rng = np.random.default_rng(random_state)
    
    # 5 Latent Archetypes (Proportions summing to 1.0)
    # 1: Agri-Allied Commercial Scalers (~20%)
    # 2: Stable Subsistence Retail / Kirana (~28%)
    # 3: Vulnerable Seasonal Producers (~18%)
    # 4: Women SHG Collective Artisans (~22%)
    # 5: Digital-First Youth Services (~12%)
    archetype_probs = [0.20, 0.28, 0.18, 0.22, 0.12]
    archetypes = [
        "Agri-Allied Commercial Scalers",
        "Stable Subsistence Retail",
        "Vulnerable Seasonal Producers",
        "Women SHG Collective Artisans",
        "Digital-First Youth Services"
    ]
    
    latent_group_idx = rng.choice(len(archetypes), size=n_samples, p=archetype_probs)
    latent_group = np.array([archetypes[i] for i in latent_group_idx])
    
    # Initial arrays
    entrepreneur_id = [f"RME_{i+1:04d}" for i in range(n_samples)]
    age = np.zeros(n_samples, dtype=int)
    gender = []
    education = []
    location_type = []
    region = []
    business_sector = []
    years_in_business = np.zeros(n_samples, dtype=float)
    loan_amount = np.zeros(n_samples, dtype=float)
    tenure_months = np.zeros(n_samples, dtype=int)
    interest_concession_pct = np.zeros(n_samples, dtype=float)
    monthly_revenue = np.zeros(n_samples, dtype=float)
    income_variability = np.zeros(n_samples, dtype=float)
    num_employees = np.zeros(n_samples, dtype=int)
    repayment_on_time_ratio = np.zeros(n_samples, dtype=float)
    num_delayed_installments = np.zeros(n_samples, dtype=int)
    has_defaulted_past = np.zeros(n_samples, dtype=int)
    mobile_banking_usage = [] # 'None', 'SMS/USSD', 'Smartphone App/UPI'
    digital_payment_ratio = np.zeros(n_samples, dtype=float)
    savings_frequency = [] # 'Never', 'Irregular', 'Monthly', 'Weekly'
    savings_balance_ratio = np.zeros(n_samples, dtype=float) # Balance / Monthly Revenue
    shg_cooperative_member = np.zeros(n_samples, dtype=int)
    market_access_type = [] # 'Local Village Only', 'Weekly Haat', 'Semi-Urban Mandi', 'Direct Institutional'
    past_training_received = np.zeros(n_samples, dtype=int)

    regions_list = ["Northern", "Southern", "Eastern", "Western", "Central"]
    region_weights = [0.20, 0.25, 0.25, 0.15, 0.15]

    for i in range(n_samples):
        group = latent_group_idx[i]
        
        # Region distribution
        reg = rng.choice(regions_list, p=region_weights)
        region.append(reg)
        
        if group == 0:
            # Agri-Allied Commercial Scalers: established, progressive, market-connected
            age[i] = int(np.clip(rng.normal(42, 7), 25, 65))
            gender.append(rng.choice(["Male", "Female"], p=[0.72, 0.28]))
            education.append(rng.choice(
                ["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate/Vocational"],
                p=[0.05, 0.15, 0.35, 0.30, 0.15]
            ))
            location_type.append(rng.choice(["Village", "Semi-Rural"], p=[0.40, 0.60]))
            business_sector.append(rng.choice(
                ["Agri-Allied", "Dairy & Livestock", "Food Processing"],
                p=[0.55, 0.30, 0.15]
            ))
            years_in_business[i] = round(float(np.clip(rng.normal(8.5, 3.5), 2.0, 25.0)), 1)
            loan_amount[i] = round(float(np.clip(rng.normal(220000, 45000), 120000, 400000)), -2)
            tenure_months[i] = rng.choice([24, 36, 48], p=[0.25, 0.50, 0.25])
            interest_concession_pct[i] = round(float(np.clip(rng.normal(3.5, 0.6), 2.0, 5.0)), 2)
            monthly_revenue[i] = round(float(np.clip(rng.normal(68000, 15000), 35000, 140000)), -2)
            income_variability[i] = round(float(np.clip(rng.beta(2, 6), 0.10, 0.40)), 3)
            num_employees[i] = int(rng.choice([1, 2, 3, 4], p=[0.30, 0.40, 0.20, 0.10]))
            repayment_on_time_ratio[i] = round(float(np.clip(rng.beta(18, 2), 0.82, 1.00)), 3)
            num_delayed_installments[i] = int(rng.poisson(0.4))
            has_defaulted_past[i] = 1 if rng.random() < 0.03 else 0
            mobile_banking_usage.append(rng.choice(["SMS/USSD", "Smartphone App/UPI"], p=[0.15, 0.85]))
            digital_payment_ratio[i] = round(float(np.clip(rng.beta(5, 3), 0.30, 0.95)), 3)
            savings_frequency.append(rng.choice(["Monthly", "Weekly"], p=[0.40, 0.60]))
            savings_balance_ratio[i] = round(float(np.clip(rng.normal(1.8, 0.5), 0.6, 4.0)), 2)
            shg_cooperative_member[i] = 1 if rng.random() < 0.85 else 0
            market_access_type.append(rng.choice(
                ["Weekly Haat", "Semi-Urban Mandi", "Direct Institutional"],
                p=[0.15, 0.55, 0.30]
            ))
            past_training_received[i] = 1 if rng.random() < 0.65 else 0
            
        elif group == 1:
            # Stable Subsistence Retail: long-running village shops, steady cash, modest aspirations
            age[i] = int(np.clip(rng.normal(46, 8), 24, 68))
            gender.append(rng.choice(["Male", "Female"], p=[0.60, 0.40]))
            education.append(rng.choice(
                ["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate/Vocational"],
                p=[0.12, 0.28, 0.40, 0.16, 0.04]
            ))
            location_type.append(rng.choice(["Village", "Semi-Rural"], p=[0.70, 0.30]))
            business_sector.append(rng.choice(
                ["Retail & Kirana", "Tailoring & Textiles", "Services & Repairs"],
                p=[0.70, 0.18, 0.12]
            ))
            years_in_business[i] = round(float(np.clip(rng.normal(10.0, 4.0), 3.0, 28.0)), 1)
            loan_amount[i] = round(float(np.clip(rng.normal(85000, 20000), 40000, 160000)), -2)
            tenure_months[i] = rng.choice([12, 18, 24, 36], p=[0.15, 0.35, 0.40, 0.10])
            interest_concession_pct[i] = round(float(np.clip(rng.normal(2.5, 0.5), 1.5, 4.0)), 2)
            monthly_revenue[i] = round(float(np.clip(rng.normal(32000, 6000), 16000, 55000)), -2)
            income_variability[i] = round(float(np.clip(rng.beta(2, 8), 0.08, 0.30)), 3)
            num_employees[i] = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
            repayment_on_time_ratio[i] = round(float(np.clip(rng.beta(14, 2.5), 0.75, 1.00)), 3)
            num_delayed_installments[i] = int(rng.poisson(0.8))
            has_defaulted_past[i] = 1 if rng.random() < 0.05 else 0
            mobile_banking_usage.append(rng.choice(["None", "SMS/USSD", "Smartphone App/UPI"], p=[0.20, 0.40, 0.40]))
            digital_payment_ratio[i] = round(float(np.clip(rng.beta(2, 4), 0.05, 0.65)), 3)
            savings_frequency.append(rng.choice(["Irregular", "Monthly", "Weekly"], p=[0.15, 0.60, 0.25]))
            savings_balance_ratio[i] = round(float(np.clip(rng.normal(1.2, 0.4), 0.3, 2.5)), 2)
            shg_cooperative_member[i] = 1 if rng.random() < 0.40 else 0
            market_access_type.append(rng.choice(
                ["Local Village Only", "Weekly Haat", "Semi-Urban Mandi"],
                p=[0.65, 0.25, 0.10]
            ))
            past_training_received[i] = 1 if rng.random() < 0.25 else 0
            
        elif group == 2:
            # Vulnerable Seasonal Producers: agricultural laborers/marginal farmers in lean periods
            age[i] = int(np.clip(rng.normal(40, 9), 20, 65))
            gender.append(rng.choice(["Male", "Female"], p=[0.55, 0.45]))
            education.append(rng.choice(
                ["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate/Vocational"],
                p=[0.35, 0.35, 0.22, 0.07, 0.01]
            ))
            location_type.append(rng.choice(["Village", "Semi-Rural"], p=[0.90, 0.10]))
            business_sector.append(rng.choice(
                ["Agri-Allied", "Dairy & Livestock", "Handicrafts & Artisan"],
                p=[0.60, 0.25, 0.15]
            ))
            years_in_business[i] = round(float(np.clip(rng.normal(5.0, 3.0), 1.0, 18.0)), 1)
            loan_amount[i] = round(float(np.clip(rng.normal(55000, 15000), 20000, 110000)), -2)
            tenure_months[i] = rng.choice([12, 18, 24], p=[0.40, 0.40, 0.20])
            interest_concession_pct[i] = round(float(np.clip(rng.normal(2.0, 0.5), 1.0, 3.5)), 2)
            monthly_revenue[i] = round(float(np.clip(rng.normal(18000, 5500), 7000, 35000)), -2)
            income_variability[i] = round(float(np.clip(rng.beta(6, 3), 0.35, 0.75)), 3)
            num_employees[i] = int(rng.choice([0, 1], p=[0.85, 0.15]))
            repayment_on_time_ratio[i] = round(float(np.clip(rng.beta(7, 5), 0.40, 0.85)), 3)
            num_delayed_installments[i] = int(rng.poisson(2.6))
            has_defaulted_past[i] = 1 if rng.random() < 0.22 else 0
            mobile_banking_usage.append(rng.choice(["None", "SMS/USSD", "Smartphone App/UPI"], p=[0.55, 0.35, 0.10]))
            digital_payment_ratio[i] = round(float(np.clip(rng.beta(1, 7), 0.0, 0.30)), 3)
            savings_frequency.append(rng.choice(["Never", "Irregular", "Monthly"], p=[0.35, 0.50, 0.15]))
            savings_balance_ratio[i] = round(float(np.clip(rng.normal(0.4, 0.2), 0.05, 1.1)), 2)
            shg_cooperative_member[i] = 1 if rng.random() < 0.45 else 0
            market_access_type.append(rng.choice(
                ["Local Village Only", "Weekly Haat"],
                p=[0.75, 0.25]
            ))
            past_training_received[i] = 1 if rng.random() < 0.12 else 0

        elif group == 3:
            # Women SHG Collective Artisans: strong group discipline, micro-scale, high repayment integrity
            age[i] = int(np.clip(rng.normal(38, 7), 21, 60))
            gender.append(rng.choice(["Female", "Male"], p=[0.94, 0.06]))
            education.append(rng.choice(
                ["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate/Vocational"],
                p=[0.20, 0.32, 0.36, 0.10, 0.02]
            ))
            location_type.append(rng.choice(["Village", "Semi-Rural"], p=[0.80, 0.20]))
            business_sector.append(rng.choice(
                ["Tailoring & Textiles", "Handicrafts & Artisan", "Food Processing", "Dairy & Livestock"],
                p=[0.40, 0.30, 0.20, 0.10]
            ))
            years_in_business[i] = round(float(np.clip(rng.normal(4.5, 2.5), 1.0, 16.0)), 1)
            loan_amount[i] = round(float(np.clip(rng.normal(45000, 12000), 20000, 85000)), -2)
            tenure_months[i] = rng.choice([12, 18, 24], p=[0.30, 0.50, 0.20])
            interest_concession_pct[i] = round(float(np.clip(rng.normal(4.0, 0.5), 2.5, 5.0)), 2)
            monthly_revenue[i] = round(float(np.clip(rng.normal(21000, 4500), 9000, 36000)), -2)
            income_variability[i] = round(float(np.clip(rng.beta(3, 5), 0.15, 0.45)), 3)
            num_employees[i] = int(rng.choice([0, 1, 2], p=[0.60, 0.30, 0.10]))
            repayment_on_time_ratio[i] = round(float(np.clip(rng.beta(20, 2), 0.88, 1.00)), 3)
            num_delayed_installments[i] = int(rng.poisson(0.3))
            has_defaulted_past[i] = 1 if rng.random() < 0.02 else 0
            mobile_banking_usage.append(rng.choice(["None", "SMS/USSD", "Smartphone App/UPI"], p=[0.20, 0.45, 0.35]))
            digital_payment_ratio[i] = round(float(np.clip(rng.beta(2, 5), 0.05, 0.50)), 3)
            savings_frequency.append(rng.choice(["Weekly", "Monthly"], p=[0.75, 0.25]))
            savings_balance_ratio[i] = round(float(np.clip(rng.normal(0.9, 0.3), 0.2, 2.0)), 2)
            shg_cooperative_member[i] = 1 if rng.random() < 0.96 else 0
            market_access_type.append(rng.choice(
                ["Local Village Only", "Weekly Haat", "Direct Institutional"],
                p=[0.50, 0.35, 0.15]
            ))
            past_training_received[i] = 1 if rng.random() < 0.55 else 0

        else: # group == 4
            # Digital-First Youth Services: younger, digitally savvy, newer, higher leverage, volatile
            age[i] = int(np.clip(rng.normal(27, 4), 19, 36))
            gender.append(rng.choice(["Male", "Female"], p=[0.70, 0.30]))
            education.append(rng.choice(
                ["Secondary", "Higher Secondary", "Graduate/Vocational"],
                p=[0.15, 0.45, 0.40]
            ))
            location_type.append(rng.choice(["Village", "Semi-Rural"], p=[0.25, 0.75]))
            business_sector.append(rng.choice(
                ["Services & Repairs", "Retail & Kirana", "Food Processing"],
                p=[0.55, 0.30, 0.15]
            ))
            years_in_business[i] = round(float(np.clip(rng.normal(2.2, 1.2), 0.5, 6.0)), 1)
            loan_amount[i] = round(float(np.clip(rng.normal(140000, 35000), 50000, 260000)), -2)
            tenure_months[i] = rng.choice([18, 24, 36], p=[0.20, 0.50, 0.30])
            interest_concession_pct[i] = round(float(np.clip(rng.normal(2.8, 0.6), 1.5, 4.5)), 2)
            monthly_revenue[i] = round(float(np.clip(rng.normal(42000, 11000), 18000, 75000)), -2)
            income_variability[i] = round(float(np.clip(rng.beta(4, 4), 0.20, 0.55)), 3)
            num_employees[i] = int(rng.choice([0, 1, 2], p=[0.40, 0.45, 0.15]))
            repayment_on_time_ratio[i] = round(float(np.clip(rng.beta(10, 3.5), 0.65, 0.95)), 3)
            num_delayed_installments[i] = int(rng.poisson(1.3))
            has_defaulted_past[i] = 1 if rng.random() < 0.09 else 0
            mobile_banking_usage.append(rng.choice(["Smartphone App/UPI"], p=[1.0]))
            digital_payment_ratio[i] = round(float(np.clip(rng.beta(6, 2), 0.50, 0.98)), 3)
            savings_frequency.append(rng.choice(["Irregular", "Monthly"], p=[0.45, 0.55]))
            savings_balance_ratio[i] = round(float(np.clip(rng.normal(0.8, 0.35), 0.15, 2.2)), 2)
            shg_cooperative_member[i] = 1 if rng.random() < 0.15 else 0
            market_access_type.append(rng.choice(
                ["Weekly Haat", "Semi-Urban Mandi", "Direct Institutional"],
                p=[0.20, 0.55, 0.25]
            ))
            past_training_received[i] = 1 if rng.random() < 0.40 else 0

    df = pd.DataFrame({
        "entrepreneur_id": entrepreneur_id,
        "age": age,
        "gender": gender,
        "education": education,
        "location_type": location_type,
        "region": region,
        "business_sector": business_sector,
        "years_in_business": years_in_business,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "interest_concession_pct": interest_concession_pct,
        "monthly_revenue": monthly_revenue,
        "income_variability": income_variability,
        "num_employees": num_employees,
        "repayment_on_time_ratio": repayment_on_time_ratio,
        "num_delayed_installments": num_delayed_installments,
        "has_defaulted_past": has_defaulted_past,
        "mobile_banking_usage": mobile_banking_usage,
        "digital_payment_ratio": digital_payment_ratio,
        "savings_frequency": savings_frequency,
        "savings_balance_ratio": savings_balance_ratio,
        "shg_cooperative_member": shg_cooperative_member,
        "market_access_type": market_access_type,
        "past_training_received": past_training_received,
        "latent_archetype_label": latent_group # Kept as benchmark reference
    })
    
    # Introduce ~1.5% heavy-tailed outliers in financial fields
    outlier_idx = rng.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    df.loc[outlier_idx, "monthly_revenue"] = df.loc[outlier_idx, "monthly_revenue"] * rng.uniform(2.5, 4.0, size=len(outlier_idx))
    df.loc[outlier_idx, "loan_amount"] = df.loc[outlier_idx, "loan_amount"] * rng.uniform(1.8, 2.8, size=len(outlier_idx))
    df["monthly_revenue"] = df["monthly_revenue"].round(-2)
    df["loan_amount"] = df["loan_amount"].round(-2)
    
    # Introduce empirical missingness (MCAR / MAR) in voluntary survey fields (3% to 5%)
    missing_spec = {
        "income_variability": 0.045,
        "savings_balance_ratio": 0.040,
        "education": 0.025,
        "past_training_received": 0.035,
        "digital_payment_ratio": 0.020
    }
    
    for col, miss_rate in missing_spec.items():
        mask = rng.random(n_samples) < miss_rate
        if col in ["education"]:
            df.loc[mask, col] = np.nan
        elif col in ["past_training_received"]:
            # nullable integer
            df[col] = df[col].astype(object)
            df.loc[mask, col] = np.nan
        else:
            df.loc[mask, col] = np.nan

    metadata = {
        "dataset_name": "MicroSegment Rural Micro-Entrepreneur Synthetic Dataset",
        "n_samples": str(n_samples),
        "random_state": str(random_state),
        "generation_method": "Multi-archetype Gaussian-Beta generative mixture with heteroskedastic noise",
        "data_status": "SYNTHETIC BENCHMARK DATASET",
        "public_data_equivalents": (
            "NABARD All India Rural Financial Inclusion Survey (NAFIS), "
            "MUDRA Annual Reports & PMMY MIS data, "
            "RBI Priority Sector Lending (PSL) & Financial Inclusion Index, "
            "World Bank Global Findex Database (India Microdata), "
            "NSSO 77th Round (Land and Livestock Holdings & Debt/Investment of Rural Households)"
        )
    }
    
    return df, metadata

if __name__ == "__main__":
    df, meta = generate_synthetic_microfinance_data(n_samples=4000, random_state=42)
    out_csv = "data/raw/rural_micro_entrepreneurs_synthetic.csv"
    df.to_csv(out_csv, index=False)
    print(f"Generated {len(df)} samples successfully -> {out_csv}")
    print("\nDataset Shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nLatent Group Distribution:")
    print(df['latent_archetype_label'].value_counts(normalize=True).round(3))
