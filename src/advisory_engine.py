"""
MicroSegment - Operational Advisory Recommendation Engine
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Transforms micro-entrepreneur profiles and segment assignments into
         actionable, persona-specific advisory cards, credit restructuring plans,
         and early warning risk evaluations for field credit officers and extension agents.
"""

import os
import json
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# 1. Operational Advisory Strategy Blueprints
# ─────────────────────────────────────────────────────────────────────

ADVISORY_STRATEGY_CATALOG = {
    0: {
        "archetype": "Agri-Allied Commercial Scalers",
        "vernacular_title": "Vikas Pragati (Commercial Producers)",
        "icon": "🚜",
        "risk_tier": "Low to Moderate",
        "touchpoint_cadence": "Monthly In-Person + Quarterly Financial Review",
        "credit_recommendation": {
            "product_name": "Kisan Pragati Term Loan & Overdraft Facility",
            "suggested_ticket_limit": "₹2,50,000 - ₹5,00,000",
            "repayment_structure": "Monthly interest servicing with quarterly principal sweeps",
            "tenure_months": 36,
            "interest_subvention_eligible": True,
            "collateral_requirement": "Asset hypothecation (machinery / equipment) + personal guarantee"
        },
        "advisory_interventions": [
            {
                "priority": 1,
                "category": "Market Linkage",
                "action": "Connect with formal off-takers and e-NAM / dairy cooperative direct procurement hubs to bypass local middlemen.",
                "timeline": "Within 30 days"
            },
            {
                "priority": 2,
                "category": "Operational Modernization",
                "action": "Facilitate equipment leasing (solar cold storage / mechanized processing) under NABARD capital investment subsidy.",
                "timeline": "Within 60 days"
            },
            {
                "priority": 3,
                "category": "Compliance & Formalization",
                "action": "Guide registration under FSSAI / Udyam and basic GST compliance to unlock priority institutional vendor contracts.",
                "timeline": "Within 90 days"
            }
        ],
        "training_curriculum": [
            "Module A3: Cold-Chain Management & Spoilage Minimization",
            "Module F2: Working Capital Ratio Optimization & Cash Flow Forecasting",
            "Module M4: Institutional B2B Contracts & Negotiation"
        ],
        "early_warning_signals": [
            "Receivables outstanding beyond 45 days from key institutional buyers.",
            "Sudden spike in cattle feed or raw material input prices (>20%).",
            "Drop in monthly power consumption indicating machinery downtime."
        ]
    },
    1: {
        "archetype": "Digital-First Youth Services",
        "vernacular_title": "Digital Yuva (Tech-Enabled Services)",
        "icon": "📱",
        "risk_tier": "Moderate (Rapid Growth, Short Tenure)",
        "touchpoint_cadence": "Bi-Weekly Digital Check-ins (WhatsApp Bot) + Quarterly Branch Visit",
        "credit_recommendation": {
            "product_name": "Yuva Udyam Flexi Digital Credit Line",
            "suggested_ticket_limit": "₹1,00,000 - ₹2,00,000",
            "repayment_structure": "Weekly auto-debit via UPI / NACH linked to transaction volume",
            "tenure_months": 18,
            "interest_subvention_eligible": True,
            "collateral_requirement": "Cashflow-based underwriting (UPI QR turnover history); Zero collateral"
        },
        "advisory_interventions": [
            {
                "priority": 1,
                "category": "Digital Expansion",
                "action": "Onboard merchant on ONDC (Open Network for Digital Commerce) and Google Business Profile to expand local reach.",
                "timeline": "Within 15 days"
            },
            {
                "priority": 2,
                "category": "Cash Flow Management",
                "action": "Introduce digital ledger tool (Khatabook/Vyapar) to prevent over-extension of customer credit balances.",
                "timeline": "Within 30 days"
            },
            {
                "priority": 3,
                "category": "Service Portfolio Diversification",
                "action": "Add banking business correspondent (BC) point / bill payment agency to stabilize daily footfall and commission income.",
                "timeline": "Within 45 days"
            }
        ],
        "training_curriculum": [
            "Module D1: Hyper-Local Digital Marketing & Social Media Sales",
            "Module F1: Daily Cash Reconciliation & Margin Accounting",
            "Module T2: Preventive Maintenance for Consumer Electronics"
        ],
        "early_warning_signals": [
            "Consecutive 3-week decline (>25%) in digital transaction volume.",
            "Excessive inventory accumulation in fast-depreciating consumer electronics.",
            "Unscheduled delays in weekly auto-debit sweeps."
        ]
    },
    2: {
        "archetype": "Vulnerable Seasonal Producers",
        "vernacular_title": "Sankat-Prone (Vulnerable Micro-Farmers)",
        "icon": "🌾",
        "risk_tier": "High / Critical Cashflow Fragility",
        "touchpoint_cadence": "Weekly In-Person BC Visits during Lean Season; Monthly Post-Harvest",
        "credit_recommendation": {
            "product_name": "Gramin Suraksha Seasonal Balloon Loan",
            "suggested_ticket_limit": "₹30,000 - ₹60,000",
            "repayment_structure": "Bullet / Balloon repayment post-harvest with 3-month moratorium during lean sowing season",
            "tenure_months": 12,
            "interest_subvention_eligible": True,
            "collateral_requirement": "SHG Solidarity Guarantee / Joint Liability Group (JLG) peer backing"
        },
        "advisory_interventions": [
            {
                "priority": 1,
                "category": "Shock Protection",
                "action": "Mandatory enrollment in Pradhan Mantri Fasal Bima Yojana (PMFBY) or weather-index micro-insurance.",
                "timeline": "Immediate (At disbursement)"
            },
            {
                "priority": 2,
                "category": "Debt Restructuring",
                "action": "Restructure existing rigid monthly amortizations into bi-annual harvest-aligned bullet installments.",
                "timeline": "Within 15 days"
            },
            {
                "priority": 3,
                "category": "Livelihood Diversification",
                "action": "Link to non-seasonal secondary micro-income streams (poultry backyard rearing, vermicompost, goat rearing).",
                "timeline": "Within 60 days"
            }
        ],
        "training_curriculum": [
            "Module A1: Climate-Resilient Cropping & Drought Mitigation",
            "Module F3: Emergency Savings Accumulation & Moneylender Escape Strategies",
            "Module L1: Secondary Livestock & Backyard Enterprise Management"
        ],
        "early_warning_signals": [
            "Monsoon deficit >20% reported in the local revenue block.",
            "Reports of pest attack or crop loss in the village cluster.",
            "Borrowing from informal local moneylenders to meet basic household consumption."
        ]
    },
    3: {
        "archetype": "Women SHG Collective Artisans",
        "vernacular_title": "Mahila Shakti (Collective Artisans)",
        "icon": "🧵",
        "risk_tier": "Very Low (Exemplary Repayment Discipline)",
        "touchpoint_cadence": "Monthly Group Meeting Attendance + Quarterly Artisan Cluster Workshop",
        "credit_recommendation": {
            "product_name": "Mahila Swavalamban Collective Artisan Credit",
            "suggested_ticket_limit": "₹50,000 - ₹1,20,000 (Group-appraised)",
            "repayment_structure": "Equal Monthly Installments (EMI) collected during monthly SHG panchayat meetings",
            "tenure_months": 24,
            "interest_subvention_eligible": True,
            "collateral_requirement": "Peer mutual guarantee via SHG / Village Organization resolution"
        },
        "advisory_interventions": [
            {
                "priority": 1,
                "category": "Intermediary Elimination",
                "action": "Aggregate collective production through Village Organization to sell directly to craft federations (TRIFED / SARAS).",
                "timeline": "Within 30 days"
            },
            {
                "priority": 2,
                "category": "Collective Bulk Procurement",
                "action": "Establish joint raw material procurement fund to reduce yarn, dyes, and packaging input costs by 20-25%.",
                "timeline": "Within 45 days"
            },
            {
                "priority": 3,
                "category": "Digital Financial Inclusion",
                "action": "Assist members in acquiring and operating UPI QR codes and BHIM Aadhaar Pay for direct customer payments.",
                "timeline": "Within 60 days"
            }
        ],
        "training_curriculum": [
            "Module C1: Contemporary Motif Design & Export-Grade Quality Standards",
            "Module D2: Digital Cataloging, Smartphone Photography & WhatsApp Business",
            "Module G1: Collective Federation Governance & Group Bookkeeping"
        ],
        "early_warning_signals": [
            "Drop in attendance or savings contributions in consecutive monthly SHG meetings.",
            "Health emergency or sudden medical expenditure in member household.",
            "Breakdown of collective marketing agreements or unsold inventory pileup."
        ]
    },
    4: {
        "archetype": "Stable Subsistence Retailers",
        "vernacular_title": "Gramin Kirana (Village Retail & Trading)",
        "icon": "🏪",
        "risk_tier": "Low (Mature & Resilient Cashflow)",
        "touchpoint_cadence": "Bi-Monthly Storefront Visits + Monthly Digital Transaction Review",
        "credit_recommendation": {
            "product_name": "Vyapar Vriddhi Revolving Stock Facility",
            "suggested_ticket_limit": "₹75,000 - ₹1,50,000",
            "repayment_structure": "Monthly EMI with automated annual renewal upon flawless servicing",
            "tenure_months": 24,
            "interest_subvention_eligible": False,
            "collateral_requirement": "Stock hypothecation + shop establishment trade license"
        },
        "advisory_interventions": [
            {
                "priority": 1,
                "category": "Payment Modernization",
                "action": "Deploy bank-sponsored UPI Soundbox with instant voice confirmation to accelerate non-cash adoption.",
                "timeline": "Within 15 days"
            },
            {
                "priority": 2,
                "category": "Distributor Direct Linkage",
                "action": "Connect retailer to FMCG super-stockists and organized B2B delivery platforms (JioBP, Udaan) to expand margins.",
                "timeline": "Within 30 days"
            },
            {
                "priority": 3,
                "category": "Credit Book Digitalization",
                "action": "Convert paper udhar khata (customer credit registry) to automated SMS payment reminder apps.",
                "timeline": "Within 45 days"
            }
        ],
        "training_curriculum": [
            "Module R1: Fast-Moving FMCG Product Assortment & Dead-Stock Minimization",
            "Module F4: Customer Credit Ceiling Management & Udhar Recovery Protocols",
            "Module D3: POS Device Operation & Loyalty Program Administration"
        ],
        "early_warning_signals": [
            "Emergence of a high-volume organized retailer or supermarket within 5km radius.",
            "Significant rise (>30%) in uncollected customer credit on the udhar ledger.",
            "Prolonged regional agricultural distress depressing overall village purchasing power."
        ]
    }
}


# ─────────────────────────────────────────────────────────────────────
# 2. Risk & Early Warning System (EWS) Evaluator
# ─────────────────────────────────────────────────────────────────────

def evaluate_early_warning_risk(row: pd.Series | dict) -> dict:
    """
    Computes a composite Early Warning Risk score (0 to 100) and severity rating
    based on cashflow stability, repayment track, leverage, and past delays.

    Robustness Guarantee: If raw applicant data is passed without pre-computed
    engineered features (e.g. from real-time demo inputs), this function automatically
    computes the exact domain features (repayment_ratio, income_stability,
    revenue_to_loan_ratio) using the standard RuralFeatureEngineer transformations,
    preventing risk understatement.
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    risk_score = 0.0
    flags = []

    # 1. Repayment Integrity (0 to 30 pts)
    # Derive repayment_ratio if not explicitly present in engineered form
    if "repayment_ratio" in row and pd.notna(row["repayment_ratio"]):
        repay = float(row["repayment_ratio"])
    elif "repayment_on_time_ratio" in row and pd.notna(row["repayment_on_time_ratio"]):
        tenure = float(row.get("tenure_months", 12)) if float(row.get("tenure_months", 12)) > 0 else 12.0
        delays = float(row.get("num_delayed_installments", 0))
        past_def = float(row.get("has_defaulted_past", 0))
        penalty = (delays / tenure) * 0.5 + (past_def * 0.25)
        repay = max(0.0, min(1.0, float(row["repayment_on_time_ratio"]) - penalty))
    else:
        repay = 0.75  # Uninformative baseline only if no repayment fields provided

    if repay < 0.50:
        risk_score += 30.0
        flags.append(f"Critical repayment erosion (effective repayment ratio: {repay:.1%})")
    elif repay < 0.70:
        risk_score += 18.0
        flags.append(f"Sub-optimal repayment discipline (effective repayment ratio: {repay:.1%})")
    elif repay < 0.85:
        risk_score += 8.0

    # 2. Income Stability (0 to 25 pts)
    # Derive income_stability if not explicitly present in engineered form
    if "income_stability" in row and pd.notna(row["income_stability"]):
        stab = float(row["income_stability"])
    elif "income_variability" in row and pd.notna(row["income_variability"]):
        stab = max(0.0, min(1.0, 1.0 - float(row["income_variability"])))
    else:
        stab = 0.65

    if stab < 0.40:
        risk_score += 25.0
        flags.append(f"Severe income volatility / climate-seasonality vulnerability (stability index: {stab:.2f})")
    elif stab < 0.60:
        risk_score += 15.0
        flags.append(f"Moderate income variability (stability index: {stab:.2f})")

    # 3. Delinquencies & Default History (0 to 25 pts)
    delays = float(row.get("num_delayed_installments", 0))
    past_default = int(row.get("has_defaulted_past", 0))
    if delays >= 3 or past_default == 1:
        risk_score += 25.0
        if delays >= 3 and past_default == 1:
            flags.append(f"Chronic past delays ({int(delays)} delayed installments) & historical default record")
        elif delays >= 3:
            flags.append(f"Chronic past delays: {int(delays)} installments missed/delayed")
        else:
            flags.append("Past loan default / 90+ DPD write-off record detected")
    elif delays >= 1:
        risk_score += 12.0
        flags.append(f"Occasional past delays: {int(delays)} installment(s) delayed")

    # 4. Debt-to-Revenue Leverage (0 to 20 pts)
    # Derive revenue_to_loan_ratio if not explicitly present in engineered form
    if "revenue_to_loan_ratio" in row and pd.notna(row["revenue_to_loan_ratio"]):
        rev_loan = float(row["revenue_to_loan_ratio"])
    elif "monthly_revenue" in row and "loan_amount" in row:
        monthly_rev = float(row.get("monthly_revenue", 0.0))
        loan = float(row.get("loan_amount", 1.0))
        loan = max(loan, 1.0)
        rev_loan = max(0.1, min(20.0, (monthly_rev * 12.0) / loan))
    else:
        rev_loan = 4.0

    if rev_loan < 1.5:
        risk_score += 20.0
        flags.append(f"High debt overhang: Annual Revenue ({rev_loan:.1f}x) is less than 1.5x requested loan")
    elif rev_loan < 2.5:
        risk_score += 10.0
        flags.append(f"Moderate debt burden: Annual Revenue ({rev_loan:.1f}x loan) below optimal 2.5x threshold")

    # Cap score
    risk_score = min(max(risk_score, 0.0), 100.0)

    # Risk Tier classification
    if risk_score >= 65.0:
        tier = "Critical Risk"
        badge_color = "red"
    elif risk_score >= 40.0:
        tier = "Moderate Risk"
        badge_color = "orange"
    elif risk_score >= 20.0:
        tier = "Low Risk"
        badge_color = "blue"
    else:
        tier = "Pristine / Minimal Risk"
        badge_color = "green"

    return {
        "risk_score": round(risk_score, 1),
        "risk_tier": tier,
        "badge_color": badge_color,
        "active_flags": flags
    }


# ─────────────────────────────────────────────────────────────────────
# 3. Comprehensive Advisory Card Generator
# ─────────────────────────────────────────────────────────────────────

def generate_advisory_card(applicant_data: dict, segment_id: int) -> dict:
    """
    Synthesizes applicant attributes, segment membership, and risk rules
    into a comprehensive, actionable Field Officer Advisory Card.
    """
    strategy = ADVISORY_STRATEGY_CATALOG.get(segment_id, ADVISORY_STRATEGY_CATALOG[4])
    risk_eval = evaluate_early_warning_risk(applicant_data)
    
    applicant_id = applicant_data.get("entrepreneur_id", "APP-UNKNOWN")
    sector = applicant_data.get("business_sector", "General")
    revenue = float(applicant_data.get("monthly_revenue", 0.0))
    loan = float(applicant_data.get("loan_amount", 0.0))
    
    card = {
        "header": {
            "applicant_id": applicant_id,
            "business_sector": sector,
            "segment_id": segment_id,
            "archetype_name": strategy["archetype"],
            "vernacular_title": strategy["vernacular_title"],
            "icon": strategy["icon"],
            "monthly_revenue": f"₹{revenue:,.0f}",
            "loan_amount": f"₹{loan:,.0f}",
            "decision_nature": "Advisory Product Recommendation & Cashflow Guidance (Non-Approval)"
        },
        "risk_assessment": risk_eval,
        "field_cadence": strategy["touchpoint_cadence"],
        "credit_blueprint": strategy["credit_recommendation"],
        "action_items": strategy["advisory_interventions"],
        "training_pathway": strategy["training_curriculum"],
        "early_warning_triggers": strategy["early_warning_signals"],
        "regulatory_disclaimer": (
            "Advisory recommendation only — Not a formal credit approval or underwriting decision. "
            "Subject to institutional KYC verification, credit bureau scrub (CIBIL/Equifax), and "
            "formal debt-service appraisal by the lending institution."
        )
    }
    
    return card


# ─────────────────────────────────────────────────────────────────────
# 4. Batch Scoring & Strategy Assignment
# ─────────────────────────────────────────────────────────────────────

def batch_assign_advisory_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments a segmented dataframe with persona titles, risk ratings,
    recommended credit products, and touchpoint frequencies.
    """
    df_out = df.copy()
    
    archetype_names = []
    vernacular_titles = []
    risk_scores = []
    risk_tiers = []
    credit_products = []
    visit_cadences = []
    
    for _, row in df_out.iterrows():
        seg_id = int(row.get("segment_label", 4))
        strat = ADVISORY_STRATEGY_CATALOG.get(seg_id, ADVISORY_STRATEGY_CATALOG[4])
        risk_info = evaluate_early_warning_risk(row)
        
        archetype_names.append(strat["archetype"])
        vernacular_titles.append(strat["vernacular_title"])
        risk_scores.append(risk_info["risk_score"])
        risk_tiers.append(risk_info["risk_tier"])
        credit_products.append(strat["credit_recommendation"]["product_name"])
        visit_cadences.append(strat["touchpoint_cadence"])
        
    df_out["segment_name"] = archetype_names
    df_out["vernacular_title"] = vernacular_titles
    df_out["ews_risk_score"] = risk_scores
    df_out["ews_risk_tier"] = risk_tiers
    df_out["recommended_product"] = credit_products
    df_out["field_cadence"] = visit_cadences
    
    return df_out


if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 6: ADVISORY RECOMMENDATION ENGINE")
    print("=" * 65)
    
    # Load segmented data
    seg_file = "data/processed/rural_micro_entrepreneurs_segmented.csv"
    if os.path.exists(seg_file):
        df_seg = pd.read_csv(seg_file)
        df_enriched = batch_assign_advisory_strategies(df_seg)
        out_path = "data/processed/rural_micro_entrepreneurs_with_advisory.csv"
        df_enriched.to_csv(out_path, index=False)
        print(f"  Enriched dataset saved -> {out_path}")
        
        # Test sample applicant card
        sample_row = df_seg.iloc[15].to_dict()
        card = generate_advisory_card(sample_row, segment_id=int(sample_row["segment_label"]))
        print("\n  Sample Field Officer Advisory Card:")
        print(f"  Applicant: {card['header']['applicant_id']} | Sector: {card['header']['business_sector']}")
        print(f"  Archetype: {card['header']['archetype_name']} ({card['header']['vernacular_title']})")
        print(f"  Risk Tier: {card['risk_assessment']['risk_tier']} (Score: {card['risk_assessment']['risk_score']}/100)")
        print(f"  Recommended Product: {card['credit_blueprint']['product_name']}")
        print(f"  Field Cadence: {card['field_cadence']}")
        print(f"  Top Action: {card['action_items'][0]['action']}")
    else:
        print(f"  [WARN] {seg_file} not found.")
        
    print("\n" + "=" * 65)
    print("PHASE 6 COMPLETE")
    print("=" * 65)
