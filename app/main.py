"""
MicroSegment - Student-Friendly Rural Micro-Entrepreneur Intelligence Platform
Course: INT 396 Capstone Project
Purpose: Interactive loan advisory and customer segmentation system for rural banking.
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.advisory_engine import ADVISORY_STRATEGY_CATALOG, evaluate_early_warning_risk
from src.classifier import RealTimeSegmentInferenceEngine


# ─────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Modern Clean Styling
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rural Business Loan Advisory System | INT 396",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Student-Friendly Styling
st.markdown("""
<style>
    /* Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #a7f3d0;
        margin-top: 2px;
    }

    /* Persona Highlight Card */
    .persona-box {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 6px solid #38bdf8;
        background-color: #1e293b;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        color: #f1f5f9;
    }
    
    /* Result Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        color: #ecfdf5;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# 2. Data & Model Caching
# ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_portfolio_data():
    csv_path = "data/processed/rural_micro_entrepreneurs_with_advisory.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/processed/rural_micro_entrepreneurs_segmented.csv"
    df = pd.read_csv(csv_path)
    return df

@st.cache_data
def load_profiles_json():
    with open("models/segment_profiles.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)
    return profiles

@st.cache_resource
def load_inference_engine():
    return RealTimeSegmentInferenceEngine()


df_portfolio = load_portfolio_data()
profiles_data = load_profiles_json()
inference_engine = load_inference_engine()

# Friendly segment naming dictionary
SEGMENT_INFO = {
    0: {"name": "Commercial Scalers", "hindi": "Vikas Pragati", "icon": "🐄", "color": "#2563eb", "desc": "Large dairy/poultry units with high turnover wanting to expand."},
    1: {"name": "Digital Youth Services", "hindi": "Digital Yuva", "icon": "📱", "color": "#8b5cf6", "desc": "Young mobile/repair shop owners doing high daily UPI volume."},
    2: {"name": "Seasonal Farmers & Producers", "hindi": "Sankat-Prone", "icon": "🌾", "color": "#ef4444", "desc": "Weather-dependent crop growers needing harvest-linked EMIs."},
    3: {"name": "Women Self-Help Artisans", "hindi": "Mahila Shakti", "icon": "🧵", "color": "#ec4899", "desc": "Women's cooperative textile/handicraft groups with 90%+ repayment."},
    4: {"name": "Village Kirana Stores", "hindi": "Gramin Kirana", "icon": "🏪", "color": "#10b981", "desc": "Daily neighborhood grocery shops needing rotating inventory loans."}
}

seg_names_map = {i: f"{SEGMENT_INFO[i]['icon']} Seg {i}: {SEGMENT_INFO[i]['name']}" for i in range(5)}
colors_map = {seg_names_map[i]: SEGMENT_INFO[i]['color'] for i in range(5)}


# ─────────────────────────────────────────────────────────────────────
# 3. Sidebar: Simple Portfolio Filter Engine
# ─────────────────────────────────────────────────────────────────────

st.sidebar.title("🌾 MicroSegment")
st.sidebar.caption("Rural Bank Loan Advisory System (INT 396)")
st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Filter Borrowers")
st.sidebar.markdown("Filter the dataset to see how loans and business categories change across India:")

# Session state reset support
if "filter_region" not in st.session_state:
    st.session_state.filter_region = "All Regions"
if "filter_sector" not in st.session_state:
    st.session_state.filter_sector = "All Sectors"
if "filter_gender" not in st.session_state:
    st.session_state.filter_gender = "All Genders"
if "filter_risk" not in st.session_state:
    st.session_state.filter_risk = "All Risk Tiers"

def reset_filters():
    st.session_state.filter_region = "All Regions"
    st.session_state.filter_sector = "All Sectors"
    st.session_state.filter_gender = "All Genders"
    st.session_state.filter_risk = "All Risk Tiers"

# Reset button
if st.sidebar.button("🔄 Reset All Filters", width="stretch"):
    reset_filters()
    st.rerun()

regions = ["All Regions"] + sorted(df_portfolio["region"].unique().tolist())
selected_region = st.sidebar.selectbox("1. Geographic Region", regions, key="filter_region")

sectors = ["All Sectors"] + sorted(df_portfolio["business_sector"].unique().tolist())
selected_sector = st.sidebar.selectbox("2. Business Sector", sectors, key="filter_sector")

genders = ["All Genders"] + sorted(df_portfolio["gender"].unique().tolist())
selected_gender = st.sidebar.selectbox("3. Gender", genders, key="filter_gender")

risk_tiers = ["All Risk Tiers"] + sorted(df_portfolio["ews_risk_tier"].dropna().unique().tolist())
selected_risk = st.sidebar.selectbox("4. Credit Risk Level", risk_tiers, key="filter_risk")

# Apply filters
filtered_df = df_portfolio.copy()
if selected_region != "All Regions":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_sector != "All Sectors":
    filtered_df = filtered_df[filtered_df["business_sector"] == selected_sector]
if selected_gender != "All Genders":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]
if selected_risk != "All Risk Tiers":
    filtered_df = filtered_df[filtered_df["ews_risk_tier"] == selected_risk]

total_borrowers = len(df_portfolio)
filtered_count = len(filtered_df)
pct_share = (filtered_count / total_borrowers) * 100

st.sidebar.markdown("---")
st.sidebar.metric(
    "Active Filtered Records",
    f"{filtered_count:,}",
    f"{pct_share:.1f}% of total 4,000"
)

st.sidebar.info(
    "💡 **Viva Tip:** Tell your teacher that traditional banks use one fixed loan rule for everyone. "
    "MicroSegment uses Machine Learning (K-Means) to group rural borrowers into 5 actionable business types."
)


# ─────────────────────────────────────────────────────────────────────
# 4. Header & DYNAMIC Top KPI Metrics
# ─────────────────────────────────────────────────────────────────────

st.title("🌾 MicroSegment: Rural Micro-Entrepreneur Loan Advisory")
st.markdown(
    "**A Smart Decision-Support System for Rural Banking** "
    "*(Intelligently segmenting micro-borrowers under MUDRA, NRLM & Regional Rural Banks)*"
)

# Calculate dynamic metrics based on filtered_df
if filtered_count > 0:
    avg_rev = filtered_df["monthly_revenue"].mean()
    avg_loan = filtered_df["loan_amount"].mean()
    avg_repay = (filtered_df["repayment_ratio"].mean()) * 100
    top_seg_id = filtered_df["segment_label"].value_counts().index[0]
    top_seg_name = SEGMENT_INFO[top_seg_id]["name"]
    top_seg_icon = SEGMENT_INFO[top_seg_id]["icon"]
    top_seg_pct = (filtered_df["segment_label"].value_counts().iloc[0] / filtered_count) * 100
else:
    avg_rev = 0
    avg_loan = 0
    avg_repay = 0
    top_seg_name = "None"
    top_seg_icon = "❌"
    top_seg_pct = 0

# Five dynamic cards that change whenever the user touches any filter
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Selected Borrowers</div>
        <div class="kpi-value">{filtered_count:,}</div>
        <div class="kpi-sub">{pct_share:.1f}% of 4,000 total</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Monthly Income</div>
        <div class="kpi-value">₹{avg_rev:,.0f}</div>
        <div class="kpi-sub">Across filtered units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Loan Amount</div>
        <div class="kpi-value">₹{avg_loan:,.0f}</div>
        <div class="kpi-sub">Sanctioned ticket size</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Repayment Integrity</div>
        <div class="kpi-value">{avg_repay:.1f}%</div>
        <div class="kpi-sub">On-time installment track</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Top Dominant Group</div>
        <div class="kpi-value" style="font-size: 1.25rem; margin-top: 8px;">{top_seg_icon} {top_seg_name}</div>
        <div class="kpi-sub">{top_seg_pct:.1f}% in this filter</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ─────────────────────────────────────────────────────────────────────
# 5. Clean, Student-Friendly Main Tabs
# ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 1. Portfolio Dashboard",
    "👥 2. The 5 Customer Groups",
    "⚡ 3. Test a Borrower (Live Demo)",
    "📈 4. How the AI Model Works",
    "🎓 5. Viva / Defense Q&A"
])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: PORTFOLIO DASHBOARD (Dynamic charts & raw data)
# ═════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📊 Dynamic Portfolio Overview")
    st.caption(f"Currently showing {filtered_count:,} borrowers based on your sidebar filter.")

    if filtered_count == 0:
        st.warning("⚠️ No borrowers match this combination of filters. Click 'Reset All Filters' in the sidebar.")
    else:
        chart_col1, chart_col2 = st.columns([1, 1.3])
        
        # Friendly segment column for plotting
        plot_df = filtered_df.copy()
        plot_df["Friendly_Segment"] = plot_df["segment_label"].map(seg_names_map)

        with chart_col1:
            seg_counts = plot_df["Friendly_Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]

            fig_donut = px.pie(
                seg_counts, names="Segment", values="Count", hole=0.5,
                color="Segment", color_discrete_map=colors_map,
                title="Customer Breakdown in Current Selection"
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_donut, width="stretch")

        with chart_col2:
            fig_scatter = px.scatter(
                plot_df,
                x="loan_amount",
                y="monthly_revenue",
                color="Friendly_Segment",
                color_discrete_map=colors_map,
                size="repayment_ratio",
                hover_data=["entrepreneur_id", "business_sector", "years_in_business", "region"],
                title="Monthly Revenue vs. Loan Amount (Bubble size = Repayment Score)",
                labels={"loan_amount": "Loan Amount (₹)", "monthly_revenue": "Monthly Revenue (₹)"}
            )
            fig_scatter.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_scatter, width="stretch")

        st.write("---")
        
        # Sector Breakdown Bar Chart
        st.subheader("🏢 Sector Distribution Across Selected Segments")
        sec_counts = (
            plot_df.groupby(["business_sector", "Friendly_Segment"])
            .size()
            .reset_index(name="count")
        )
        sec_totals = sec_counts.groupby("business_sector")["count"].transform("sum")
        sec_counts["percentage"] = (sec_counts["count"] / sec_totals * 100).round(1)

        fig_sec = px.bar(
            sec_counts,
            x="business_sector",
            y="percentage",
            color="Friendly_Segment",
            color_discrete_map=colors_map,
            title="Percentage Share of Each Segment by Business Sector (%)",
            labels={"percentage": "Share (%)", "business_sector": "Business Sector", "Friendly_Segment": "Group"}
        )
        fig_sec.update_layout(barmode="stack", xaxis_tickangle=-20, margin=dict(t=40, b=40, l=10, r=10), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_sec, width="stretch")

        st.write("---")
        
        # Interactive Searchable Borrower Table
        st.subheader("📋 Inspect Calibrated Synthetic Borrower Records")
        st.caption("Explore synthetic borrower sample records (N=4,000) calibrated against RBI, NABARD (NAFIS), and MUDRA distribution benchmarks:")
        
        display_cols = [
            "entrepreneur_id", "gender", "age", "region", "business_sector",
            "monthly_revenue", "loan_amount", "repayment_ratio", "segment_name",
            "ews_risk_tier", "recommended_product"
        ]
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols].rename(columns={
                "entrepreneur_id": "ID", "monthly_revenue": "Revenue (₹)",
                "loan_amount": "Loan (₹)", "repayment_ratio": "Repayment Track",
                "segment_name": "Assigned Segment", "ews_risk_tier": "Risk Level",
                "recommended_product": "Tailored Loan Product"
            }).head(100),
            width="stretch",
            height=280
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 2: THE 5 CUSTOMER GROUPS (Plain English Personas)
# ═════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("👥 The 5 Discovered Rural Business Personas")
    st.markdown(
        "Instead of treating all rural borrowers the same, our K-Means clustering algorithm "
        "discovered **5 natural business profiles** in the rural economy:"
    )

    # 5 Summary cards side-by-side
    p_cols = st.columns(5)
    for i, col in enumerate(p_cols):
        info = SEGMENT_INFO[i]
        with col:
            st.markdown(f"""
            <div style="background-color: #1e293b; border-top: 4px solid {info['color']}; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #f8fafc;">{info['icon']} Group {i}</h4>
                <b style="color: {info['color']};">{info['name']}</b><br>
                <i style="color: #94a3b8; font-size: 0.85rem;">({info['hindi']})</i>
                <p style="font-size: 0.82rem; color: #cbd5e1; margin-top: 8px;">{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("---")
    st.subheader("🔍 Inspect Detailed Loan Product & Advice for Any Group")

    selected_group_id = st.selectbox(
        "Select a Group to View Its Custom Bank Policy:",
        options=list(range(5)),
        format_func=lambda x: f"Group {x}: {SEGMENT_INFO[x]['icon']} {SEGMENT_INFO[x]['name']} ({SEGMENT_INFO[x]['hindi']})"
    )

    strat = ADVISORY_STRATEGY_CATALOG[selected_group_id]
    p_meta = profiles_data[str(selected_group_id)]
    cb = strat["credit_recommendation"]

    # Main details container
    st.markdown(f"""
    <div class="persona-box" style="border-left-color: {SEGMENT_INFO[selected_group_id]['color']};">
        <h3>{SEGMENT_INFO[selected_group_id]['icon']} Group {selected_group_id}: {strat['archetype']} — <i>{strat['vernacular_title']}</i></h3>
        <p style="font-size: 1rem; color: #94a3b8;"><b>Target Need:</b> {p_meta['tagline']}</p>
        <span style="background-color: #0369a1; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">Risk Tier: {strat['risk_tier']}</span>
        &nbsp;&nbsp;
        <span style="background-color: #334155; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">Portfolio Share: {p_meta['percentage_of_portfolio']}% ({p_meta['size']:,} businesses)</span>
    </div>
    """, unsafe_allow_html=True)

    col_prod, col_advice = st.columns(2)
    with col_prod:
        st.write("#### 💳 Tailored Loan Recommendation")
        st.write(f"• **Recommended Product:** `{cb['product_name']}`")
        st.write(f"• **Recommended Loan Limit:** **{cb['suggested_ticket_limit']}**")
        st.write(f"• **Repayment Plan:** {cb['repayment_structure']}")
        st.write(f"• **Loan Tenure:** {cb['tenure_months']} Months")
        st.write(f"• **Collateral Needed:** {cb['collateral_requirement']}")
        st.write(f"• **Government Subvention:** {'✅ Eligible for 4% Interest Subvention' if cb['interest_subvention_eligible'] else '❌ Standard Bank Interest'}")

    with col_advice:
        st.write("#### 📋 Field Officer Action Checklist")
        st.write(f"• **Visit Schedule:** `{strat['touchpoint_cadence']}`")
        for act in strat["advisory_interventions"]:
            st.info(f"**Priority {act['priority']} ({act['category']}):** {act['action']}")
        
        st.write("#### ⚠️ Early Warning Signs to Watch For")
        for trig in strat["early_warning_signals"]:
            st.warning(f"• {trig}")


# ═════════════════════════════════════════════════════════════════════
# TAB 3: TEST A BORROWER (Interactive Scoring & Advisory Demo)
# ═════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("⚡ Live Demo: Test a Synthetic Loan Applicant Profile")
    st.markdown(
        "Try this in your viva! Choose a calibrated demo borrower profile or enter custom values. "
        "Our trained **Random Forest cluster partition emulator** categorizes the applicant, computes domain-engineered risk metrics, "
        "and generates a personalized, actionable advisory action card."
    )

    # Preset synthetic applicant library with complete attribute definitions
    presets = {
        "Preset 1: Ramesh Dairy (Agri-Allied Commercial Scaler)": {
            "age": 42, "gender": "Male", "education": "Secondary", "location_type": "Rural", "region": "North",
            "business_sector": "Agri-Allied & Dairy", "years_in_business": 9.0, "loan_amount": 220000.0,
            "tenure_months": 36, "interest_concession_pct": 3.0, "monthly_revenue": 72000.0,
            "income_variability": 0.22, "num_employees": 2, "repayment_on_time_ratio": 0.95,
            "num_delayed_installments": 0, "has_defaulted_past": 0, "mobile_banking_usage": "Smartphone App/UPI",
            "digital_payment_ratio": 0.55, "savings_frequency": "Weekly", "savings_balance_ratio": 0.35,
            "shg_cooperative_member": 1, "market_access_type": "Wholesale Mandi", "past_training_received": 1
        },
        "Preset 2: Ankit Mobile Services (Digital-First Youth Services)": {
            "age": 24, "gender": "Male", "education": "Graduate", "location_type": "Semi-Urban", "region": "South",
            "business_sector": "Services & Repair", "years_in_business": 2.0, "loan_amount": 140000.0,
            "tenure_months": 24, "interest_concession_pct": 2.5, "monthly_revenue": 45000.0,
            "income_variability": 0.45, "num_employees": 1, "repayment_on_time_ratio": 0.85,
            "num_delayed_installments": 1, "has_defaulted_past": 0, "mobile_banking_usage": "Smartphone App/UPI",
            "digital_payment_ratio": 0.88, "savings_frequency": "Irregular", "savings_balance_ratio": 0.15,
            "shg_cooperative_member": 0, "market_access_type": "Direct Consumer", "past_training_received": 1
        },
        "Preset 3: Birbal Farmer (Vulnerable Rainfed Crop Farmer)": {
            "age": 48, "gender": "Male", "education": "Primary", "location_type": "Remote Rural", "region": "East",
            "business_sector": "Crop Farming & Cultivation", "years_in_business": 5.0, "loan_amount": 50000.0,
            "tenure_months": 12, "interest_concession_pct": 4.0, "monthly_revenue": 16000.0,
            "income_variability": 0.68, "num_employees": 0, "repayment_on_time_ratio": 0.60,
            "num_delayed_installments": 3, "has_defaulted_past": 1, "mobile_banking_usage": "None",
            "digital_payment_ratio": 0.05, "savings_frequency": "Never", "savings_balance_ratio": 0.02,
            "shg_cooperative_member": 0, "market_access_type": "Local Haat", "past_training_received": 0
        },
        "Preset 4: Sunita Devi Weaver (Women Self-Help Group Artisan)": {
            "age": 36, "gender": "Female", "education": "Secondary", "location_type": "Rural", "region": "Central",
            "business_sector": "Handicrafts & Artisans", "years_in_business": 4.5, "loan_amount": 45000.0,
            "tenure_months": 24, "interest_concession_pct": 3.5, "monthly_revenue": 21000.0,
            "income_variability": 0.32, "num_employees": 0, "repayment_on_time_ratio": 0.98,
            "num_delayed_installments": 0, "has_defaulted_past": 0, "mobile_banking_usage": "Smartphone App/UPI",
            "digital_payment_ratio": 0.35, "savings_frequency": "Weekly", "savings_balance_ratio": 0.85,
            "shg_cooperative_member": 1, "market_access_type": "Artisan Cluster", "past_training_received": 1
        },
        "Preset 5: Suresh Kirana (Stable Subsistence Retailer)": {
            "age": 45, "gender": "Male", "education": "Secondary", "location_type": "Rural", "region": "Central",
            "business_sector": "Retail & Kirana", "years_in_business": 11.0, "loan_amount": 85000.0,
            "tenure_months": 24, "interest_concession_pct": 2.0, "monthly_revenue": 33000.0,
            "income_variability": 0.28, "num_employees": 0, "repayment_on_time_ratio": 0.88,
            "num_delayed_installments": 1, "has_defaulted_past": 0, "mobile_banking_usage": "Smartphone App/UPI",
            "digital_payment_ratio": 0.32, "savings_frequency": "Monthly", "savings_balance_ratio": 0.40,
            "shg_cooperative_member": 0, "market_access_type": "Local Haat", "past_training_received": 0
        },
        "Custom Input (Enter your own values)": {
            "age": 35, "gender": "Female", "education": "Secondary", "location_type": "Rural", "region": "Central",
            "business_sector": "Retail & Kirana", "years_in_business": 4.0, "loan_amount": 75000.0,
            "tenure_months": 24, "interest_concession_pct": 2.5, "monthly_revenue": 30000.0,
            "income_variability": 0.30, "num_employees": 0, "repayment_on_time_ratio": 0.85,
            "num_delayed_installments": 0, "has_defaulted_past": 0, "mobile_banking_usage": "Smartphone App/UPI",
            "digital_payment_ratio": 0.30, "savings_frequency": "Monthly", "savings_balance_ratio": 0.25,
            "shg_cooperative_member": 0, "market_access_type": "Direct Consumer", "past_training_received": 0
        }
    }

    selected_preset_name = st.selectbox("Choose a Synthetic Borrower Profile for Instant Demo:", list(presets.keys()))
    is_custom = selected_preset_name.startswith("Custom")
    active_data = presets[selected_preset_name]

    st.write("---")
    st.write("#### ✏️ Core Financial & Operational Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_rev = st.number_input("Monthly Income / Revenue (₹)", min_value=5000.0, max_value=500000.0, value=float(active_data["monthly_revenue"]), step=2000.0)
        loan_amt = st.number_input("Requested Loan Amount (₹)", min_value=10000.0, max_value=1000000.0, value=float(active_data["loan_amount"]), step=5000.0)
        business_sector = st.selectbox("Business Sector", sectors[1:], index=sectors[1:].index(active_data["business_sector"]) if active_data["business_sector"] in sectors[1:] else 0)

    with c2:
        repay_score = st.slider("On-Time Repayment History (%)", min_value=0.0, max_value=100.0, value=float(active_data["repayment_on_time_ratio"] * 100)) / 100.0
        upi_ratio = st.slider("UPI / Digital Collections Ratio (%)", min_value=0.0, max_value=100.0, value=float(active_data["digital_payment_ratio"] * 100)) / 100.0
        delayed_installments = st.number_input("Past Delayed Installments Count", min_value=0, max_value=12, value=int(active_data["num_delayed_installments"]))

    with c3:
        years_biz = st.number_input("Years in Business", min_value=0.5, max_value=30.0, value=float(active_data["years_in_business"]))
        num_emp = st.number_input("Number of Non-Owner Employees", min_value=0, max_value=20, value=int(active_data["num_employees"]))
        is_shg = st.radio("SHG / Collective Member?", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0], index=0 if active_data["shg_cooperative_member"] == 1 else 1)[1]

    # Advanced and demographic input controls to guarantee complete transparency
    with st.expander("🛠️ Full Model Feature Controls (Inspect / Customize All 23 Inputs)", expanded=is_custom):
        st.caption("When using Custom Input, modify any demographic, loan terms, or banking parameters below:")
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            inp_age = st.number_input("Borrower Age", min_value=18, max_value=75, value=int(active_data.get("age", 35)))
            inp_gender = st.selectbox("Gender", ["Female", "Male"], index=0 if active_data.get("gender") == "Female" else 1)
            inp_edu = st.selectbox("Education Level", ["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate"],
                                   index=["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate"].index(active_data.get("education", "Secondary")) if active_data.get("education") in ["No Formal", "Primary", "Secondary", "Higher Secondary", "Graduate"] else 2)
            inp_loc = st.selectbox("Location Type", ["Village", "Semi-Rural", "Remote Rural"], index=0)
        with adv2:
            inp_region = st.selectbox("Region", ["Central", "North", "South", "East", "West"],
                                      index=["Central", "North", "South", "East", "West"].index(active_data.get("region", "Central")) if active_data.get("region") in ["Central", "North", "South", "East", "West"] else 0)
            inp_market = st.selectbox("Market Access Channel", ["Direct Consumer", "Local Haat", "Wholesale Mandi", "Artisan Cluster", "Direct Institutional"],
                                      index=["Direct Consumer", "Local Haat", "Wholesale Mandi", "Artisan Cluster", "Direct Institutional"].index(active_data.get("market_access_type", "Direct Consumer")) if active_data.get("market_access_type") in ["Direct Consumer", "Local Haat", "Wholesale Mandi", "Artisan Cluster", "Direct Institutional"] else 0)
            inp_tenure = st.selectbox("Loan Tenure (Months)", [12, 18, 24, 36, 48],
                                      index=[12, 18, 24, 36, 48].index(active_data.get("tenure_months", 24)) if active_data.get("tenure_months") in [12, 18, 24, 36, 48] else 2)
            inp_concession = st.slider("Interest Subvention Concession (%)", min_value=0.0, max_value=5.0, value=float(active_data.get("interest_concession_pct", 3.0)), step=0.5)
        with adv3:
            inp_variability = st.slider("Income Seasonality / Variability (CV)", min_value=0.05, max_value=0.85, value=float(active_data.get("income_variability", 0.30)), step=0.05)
            inp_past_def = st.radio("Has Past Default / Write-Off?", [("No Past Default", 0), ("Has Defaulted in Past", 1)], format_func=lambda x: x[0],
                                    index=int(active_data.get("has_defaulted_past", 0)))[1]
            inp_savings_freq = st.selectbox("Formal / SHG Savings Frequency", ["Never", "Irregular", "Monthly", "Weekly"],
                                            index=["Never", "Irregular", "Monthly", "Weekly"].index(active_data.get("savings_frequency", "Weekly")) if active_data.get("savings_frequency") in ["Never", "Irregular", "Monthly", "Weekly"] else 2)
            inp_savings_bal = st.slider("Savings Balance to Monthly Revenue Ratio", min_value=0.0, max_value=4.0, value=float(active_data.get("savings_balance_ratio", 0.30)), step=0.05)

    # Predict button
    if st.button("⚡ Evaluate Borrower & Generate Advisory Recommendation", type="primary", width="stretch"):
        applicant_payload = {
            "age": int(inp_age),
            "gender": str(inp_gender),
            "education": str(inp_edu),
            "location_type": str(inp_loc),
            "region": str(inp_region),
            "business_sector": str(business_sector),
            "years_in_business": float(years_biz),
            "loan_amount": float(loan_amt),
            "tenure_months": int(inp_tenure),
            "interest_concession_pct": float(inp_concession),
            "monthly_revenue": float(monthly_rev),
            "income_variability": float(inp_variability),
            "num_employees": int(num_emp),
            "repayment_on_time_ratio": float(repay_score),
            "num_delayed_installments": int(delayed_installments),
            "has_defaulted_past": int(inp_past_def),
            "mobile_banking_usage": "Smartphone App/UPI" if upi_ratio > 0.3 else ("SMS/USSD" if upi_ratio > 0.05 else "None"),
            "digital_payment_ratio": float(upi_ratio),
            "savings_frequency": str(inp_savings_freq),
            "savings_balance_ratio": float(inp_savings_bal),
            "shg_cooperative_member": int(is_shg),
            "market_access_type": str(inp_market),
            "past_training_received": int(active_data.get("past_training_received", 0))
        }

        with st.spinner("AI evaluating borrower archetype and computing risk indicators..."):
            res = inference_engine.predict(applicant_payload)
            pred_id = res["predicted_segment_id"]
            conf = res["confidence"] * 100
            card = res["advisory_card"]
            risk = card["risk_assessment"]  # Utilizes exact engineered features
            probs = res.get("probabilities_by_segment_id", {})
            enriched = res.get("enriched_applicant", {})

        st.success("✅ Operational Assessment Complete!")

        # Non-Approval Regulatory Disclaimer
        st.warning(
            "⚖️ **Regulatory & Underwriting Disclaimer:** This system provides advisory product matching and cashflow structuring "
            "recommendations based on unsupervised borrower archetype modeling. It does **not** constitute a formal loan sanction or underwriting credit approval. "
            "Formal credit decisions require institutional KYC verification, credit bureau appraisal (CIBIL/Equifax), and debt-service capacity verification."
        )

        # Recommendation result container
        st.markdown(f"""
        <div class="recommendation-box">
            <h2 style="margin: 0; color: #a7f3d0;">{SEGMENT_INFO[pred_id]['icon']} Categorized As: Group {pred_id} — {SEGMENT_INFO[pred_id]['name']}</h2>
            <p style="font-size: 1.1rem; margin-top: 5px;">Vernacular: <b>{SEGMENT_INFO[pred_id]['hindi']}</b> &nbsp;|&nbsp; Model Confidence: <b>{conf:.1f}%</b></p>
            <hr style="border-color: #059669;">
            <p style="font-size: 1.1rem; margin-bottom: 5px;">💳 <b>Suggested Product (Advisory Recommendation):</b> {card['credit_blueprint']['product_name']}</p>
            <p style="font-size: 1.1rem; margin-bottom: 5px;">💰 <b>Recommended Ticket Size:</b> {card['credit_blueprint']['suggested_ticket_limit']}</p>
            <p style="font-size: 1.1rem; margin-bottom: 5px;">📅 <b>Suggested Repayment Structure:</b> {card['credit_blueprint']['repayment_structure']}</p>
            <p style="font-size: 1.1rem;">🏃 <b>Officer Touchpoint Cadence:</b> {card['field_cadence']}</p>
        </div>
        """, unsafe_allow_html=True)

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.write("#### 🎯 Immediate Field Officer Actions")
            for act in card["action_items"]:
                st.info(f"• **Priority {act['priority']} ({act['category']}):** {act['action']}")
        with res_c2:
            st.write(f"#### ⚠️ Early Warning Risk: `{risk['risk_tier']}` (Score: {risk['risk_score']}/100)")
            if risk["active_flags"]:
                for fl in risk["active_flags"]:
                    st.warning(f"🚩 {fl}")
            else:
                st.success("✅ Clean financial profile! Zero active risk flags.")

            # View computed engineered features expander
            with st.expander("🔍 View Computed Engineered Indicators (Passed to EWS)"):
                st.write(f"• **Effective Repayment Ratio:** `{enriched.get('repayment_ratio', 0.0):.1%}`")
                st.write(f"• **Income Stability Index:** `{enriched.get('income_stability', 0.0):.2f}` (1.0 = stable)")
                st.write(f"• **Annual Revenue to Loan Ratio:** `{enriched.get('revenue_to_loan_ratio', 0.0):.2f}x`")
                st.write(f"• **Digital Adoption Score:** `{enriched.get('digital_adoption_score', 0.0):.2f}`")
                st.write(f"• **Business Maturity Index:** `{enriched.get('business_maturity', 0.0):.2f}`")

        # Class Probability Distribution Breakdown Chart
        st.write("---")
        st.write("#### 📊 Prediction Confidence Across All 5 Segments")
        prob_data = []
        for seg_idx in range(5):
            prob_val = probs.get(seg_idx, 0.0) * 100
            prob_data.append({
                "Segment": f"{SEGMENT_INFO[seg_idx]['icon']} Seg {seg_idx}: {SEGMENT_INFO[seg_idx]['name']}",
                "Probability": prob_val,
                "Color": SEGMENT_INFO[seg_idx]["color"]
            })
        df_probs = pd.DataFrame(prob_data)
        fig_probs = px.bar(
            df_probs, y="Segment", x="Probability", orientation="h",
            text=df_probs["Probability"].apply(lambda v: f"{v:.1f}%"),
            title="Surrogate Classifier Confidence Distribution Across Borrower Archetypes (%)",
            color="Segment",
            color_discrete_map={d["Segment"]: d["Color"] for d in prob_data}
        )
        fig_probs.update_layout(showlegend=False, xaxis_range=[0, 100], margin=dict(t=40, b=10, l=10, r=10), height=230)
        st.plotly_chart(fig_probs, width="stretch")

        # One-Click JSON Download of Official Action Card
        card_json = json.dumps(card, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Official Field Officer Action Card (JSON)",
            data=card_json,
            file_name=f"Field_Officer_Action_Card_Group_{pred_id}.json",
            mime="application/json",
            width="stretch"
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 4: HOW THE AI MODEL WORKS (Methodology & Validation)
# ═════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📈 How the Machine Learning Model Works")
    st.markdown("""
    This project uses a 2-stage Machine Learning architecture:
    1. **Unsupervised Learning (K-Means Clustering):** Discovers 5 natural borrower archetypes from 4,000 multi-dimensional rural business profiles without requiring pre-assigned human labels.
    2. **Supervised Surrogate Classifier (Random Forest):** Emulates the discovered cluster boundaries with **98.8% partition recovery fidelity (5-Fold CV Macro F1)**, enabling instantaneous millisecond scoring for new loan applicants without re-computing pairwise distance matrices.
    """)

    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        st.write("#### 1. Cluster Visualization (2D PCA & UMAP Projections)")
        st.image("reports/figures/pca_umap_clusters.png", caption="2D PCA & UMAP qualitative projections of the 32-dimensional feature space", width="stretch")
        st.caption(
            "Note: 2D projections provide qualitative geometric intuition of the multi-dimensional feature manifold. "
            "Dimensionality reduction can compress or expand local distances; quantitative validation is carried out via mathematical metrics below."
        )

    with ai_col2:
        st.write(r"#### 2. Model Selection & Metric Sweep ($k=2 \dots 10$)")
        st.image("reports/figures/cluster_metric_sweep.png", caption="Validation metrics across k=2 to k=10 (Silhouette, Davies-Bouldin, Calinski-Harabasz, Inertia, BIC)", width="stretch")
        st.caption(
            "Silhouette score achieves its global peak at k=5 (0.2515, a +21.5% boost over k=4), "
            "while Davies-Bouldin achieves its global minimum (1.4442). This establishes k=5 as the mathematical global optimum."
        )

    st.write("---")
    ai_col3, ai_col4 = st.columns(2)
    with ai_col3:
        st.write("#### 3. Surrogate Classifier Partition Emulation (Confusion Matrix)")
        st.image("reports/figures/classifier_confusion_matrix.png", caption="5-Fold Stratified Cross-Validation Confusion Matrix (Partition Recovery)", width="stretch")
        st.caption(
            "The 98.8% Macro F1 score demonstrates high fidelity in emulating the K-Means cluster partition. "
            "(Important: This evaluates cluster emulation fidelity, not real-world credit default prediction; prospective deployment requires empirical field validation)."
        )

    with ai_col4:
        st.write("#### 4. Cluster Stability Test (Bootstrap ARI)")
        st.image("reports/figures/cluster_stability.png", caption="Bootstrap Resampling Distribution (B=100 iterations, 80% subsampling, Mean ARI = 0.941)", width="stretch")
        st.caption(
            "A Mean Adjusted Rand Index (ARI) of 0.941 ± 0.038 across 100 perturbed subsamples confirms that cluster boundaries are structurally persistent and resistant to sampling noise."
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 5: VIVA & DEFENSE Q&A (Academic & Faculty Defense Guide)
# ═════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🎓 Viva & Defense Guide: Questions Your Professor Will Ask")
    st.markdown("Clear, statistically rigorous answers for academic evaluation and faculty viva defense:")

    with st.expander("Q1: What is the main problem and objective of your project?"):
        st.markdown("""
        **Core Defense Answer:**
        > *"In rural priority-sector lending (MUDRA, NRLM, Regional Rural Banks), small micro-enterprises are treated as a **monolithic, undifferentiated cohort**. 
        A rainfed seasonal crop farmer receives the exact same rigid monthly EMI as a village grocery retailer, causing preventable defaults and informal debt traps. 
        Conversely, productive growing units receive suboptimal credit limits and zero market linkage.
        
        **MicroSegment** uses unsupervised machine learning to discover 5 structurally distinct socio-economic archetypes at loan origination, 
        and operationalizes them into persona-specific advisory cards, flexible repayment blueprints, and proactive early warning risk alerts."*
        """)

    with st.expander("Q2: What do the accuracy figures (98.8% F1) actually mean? Does this prove real-world accuracy?"):
        st.markdown("""
        **Statistical Clarification for Examiner:**
        > *"The 98.8% cross-validation Macro F1-score measures **Cluster Partition Emulation Fidelity**, NOT real-world borrower credit prediction accuracy.
        
        Here is the critical distinction:
        1. **Unsupervised Discovery:** K-Means discovered the 5 natural customer clusters from the data without labels.
        2. **Surrogate Classifier:** To score walk-in applicants in milliseconds without re-computing $N=4,000$ distance matrices, we trained a Random Forest emulator. The 98.8% F1 confirms that the Random Forest reproduces K-Means decisions with near-perfect fidelity.
        3. **Synthetic Generator Recovery:** Comparing clustering against the generator's latent labels yields an ARI of 0.982, confirming recovery of the synthetic structure.
        4. **Real-World Validation Requirement:** To make production claims for actual bank disbursement, prospective validation must be conducted on independent empirical administrative data (e.g. NABARD NAFIS or MUDRA loan books)."*
        """)

    with st.expander("Q3: Your initial framing target was Silhouette ≥ 0.35 and DB ≤ 1.35, but the model scored 0.2515 and 1.4442. Did the model fail?"):
        st.markdown(r"""
        **Methodological Justification for Examiner:**
        > *"No, the model did not fail. Initial targets of ≥ 0.35 were hypothetical textbook benchmarks derived from low-dimensional synthetic Gaussian blobs. In real-world and high-dimensional survey datasets (32 continuous and one-hot features):
        
        1. **Curse of Dimensionality:** Euclidean distance concentration in 32 dimensions naturally lowers geometric silhouette width.
        2. **Continuous Socio-Economic Overlap:** Viable rural businesses exist along a continuous spectrum rather than in isolated hyper-spheres; forcing artificial separation would distort reality.
        3. **Global Mathematical Optimum:** Across all tested algorithms and cluster counts ($k=2 \dots 10$), **$k=5$ is the unambiguous simultaneous global optimum**:
           - Peak Silhouette ($0.2515$, a $+21.5\%$ jump over $k=4$).
           - Minimum Davies-Bouldin ($1.4442$, lower is better).
           - Peak Calinski-Harabasz ($1024.1$).
        4. **Exceptional Stability:** The Bootstrap stability test achieved a **Mean ARI of 0.941 ± 0.038** across 100 resamples (well above the academic robustness threshold of 0.80), proving the 5 clusters are structurally genuine and mathematically resilient."*
        """)

    with st.expander("Q4: Are demographic attributes (gender, education, region, SHG) included? What are the fair lending safeguards?"):
        st.markdown("""
        **Fair Lending & Ethical Defense:**
        > *"Our model explicitly differentiates between prohibited social attributes and operational socio-demographic channels:
        
        1. **Strictly Excluded Attributes:** Caste, religion, marital status, and personal familial assets were strictly omitted to prevent systemic bias and unlawful discrimination.
        2. **Operationally Included Attributes:**
           - **Gender & SHG Membership:** Essential in Indian priority-sector lending because the National Rural Livelihood Mission (NRLM) operates primarily through women's Self-Help Groups (SHGs). Including these attributes allows the system to direct affirmative 4% interest subventions and collective marketing access to women artisans (Segment 3).
           - **Region & Location Type:** Captures agro-climatic seasonality, crop cycles, and market density across India.
           - **Education:** Used as an operational proxy to tailor extension training formats (vernacular/visual vs digital accounting).
        3. **Affirmative Support vs. Credit Denial:** Segment assignment is used **exclusively for tailored advisory, cashflow structuring, and positive enablement — never for automated credit denial, loan rejection, or predatory risk pricing**."*
        """)

    with st.expander("Q5: The dataset is synthetic. What are its limitations, and how would you validate it empirically?"):
        st.markdown("""
        **Data Science Realism Answer:**
        > *"The 4,000 records were generated using a calibrated multi-archetype mixture model reflecting RBI, NABARD (NAFIS), and MUDRA distribution parameters, including realistic right-skewness and missingness (MCAR).
        
        However, synthetic data has inherent limitations:
        - It assumes parametric distributions that may underestimate macroeconomic co-dependencies (e.g. state-wide droughts or supply chain disruptions).
        - To validate this architecture empirically before production deployment, the pipeline can directly ingest plug-and-play public datasets:
          1. **NABARD All India Rural Financial Inclusion Survey (NAFIS)**
          2. **MUDRA Scheme Cohort Performance Reports (PMMY MIS)**
          3. **RBI Priority Sector Lending (PSL) District Credit Data**
          4. **NSSO 77th Round (Schedule 18.2: Debt & Investment)**"*
        """)

    with st.expander("Q6: Why does the system output 'Suggested Products' instead of an approved loan decision?"):
        st.markdown("""
        **Operational Boundary Answer:**
        > *"In institutional banking, automated algorithmic loan approvals without human oversight introduce severe regulatory and credit risks. 
        
        **MicroSegment is a Decision-Support System, not an Automated Underwriter:**
        - It recommends the *optimal product archetype and cashflow structure* (e.g., harvest-aligned bullet repayment vs. monthly EMI) based on behavioral clustering.
        - Formal credit sanction remains subject to human loan officer review, KYC compliance, debt-service capacity appraisal, and institutional credit bureau checks (CIBIL/Equifax)."*
        """)

    with st.expander("Q7: Why did you apply the Yeo-Johnson Power Transformation before clustering?"):
        st.markdown("""
        **Mathematical Rationale:**
        > *"Rural enterprise financials exhibit extreme right-skewness (monthly revenue skewness was $+2.78$). In standard Euclidean distance calculations used by K-Means, large revenue values would completely dominate all other features.
        
        We selected **Yeo-Johnson transformation** over Box-Cox because it supports zero and negative values (common in net margins and delay counts). It successfully normalized revenue skewness from $+2.78 \to 0.006$ via maximum likelihood estimation (MLE), ensuring balanced distance computations."*
        """)

st.markdown("---")
st.caption("MicroSegment | Academic Capstone Project (INT 396) | Python, Scikit-Learn & Streamlit | Calibrated Synthetic Portfolio")
