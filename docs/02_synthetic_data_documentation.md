# Phase 2: Synthetic Data Generation & Data Dictionary

> [!NOTE]
> **Data Label:** SYNTHETIC BENCHMARK DATASET  
> This dataset was generated via a multi-archetype mixture model (`seed=42`) reflecting socio-economic dynamics of priority-sector concessional micro-borrowers in rural India. No real customer PII is utilized.

---

## 1. Dataset Overview
- **Total Records:** 4,000 micro-entrepreneurs
- **Feature Count:** 24 operational and demographic attributes + 1 latent ground-truth benchmark label
- **Generated File Location:** `data/raw/rural_micro_entrepreneurs_synthetic.csv`
- **Generator Script:** `src/data_generator.py`

---

## 2. Complete Data Dictionary

| Variable | Type | Range / Categories | Description & Socio-Economic Rationale |
| :--- | :--- | :--- | :--- |
| `entrepreneur_id` | String | `RME_0001` - `RME_4000` | Unique borrower reference identifier |
| `age` | Integer | 19 – 68 years | Age of primary business owner |
| `gender` | Categorical | Male (55.4%), Female (44.6%) | Gender split reflective of rural credit channels (MFI / SHG / MUDRA) |
| `education` | Categorical | No Formal, Primary, Secondary, Higher Secondary, Graduate/Vocational | Highest educational attainment (2.1% missing) |
| `location_type` | Categorical | Village (68.4%), Semi-Rural (31.6%) | Geographic tier of operating premises |
| `region` | Categorical | Northern, Southern, Eastern, Western, Central | Macro administrative zone in India |
| `business_sector` | Categorical | Agri-Allied, Dairy & Livestock, Retail & Kirana, Tailoring & Textiles, Handicrafts & Artisan, Food Processing, Services & Repairs | Primary economic revenue driver |
| `years_in_business` | Float | 0.5 – 23.3 years | Operating tenure / enterprise vintage |
| `loan_amount` | Float | ₹20,000 – ₹609,800 | Concessional loan principal disbursed (INR) |
| `tenure_months` | Integer | 12, 18, 24, 36, 48 | Loan repayment horizon in months |
| `interest_concession_pct`| Float | 1.00% – 5.00% | Interest rate subvention below commercial MFI baseline |
| `monthly_revenue` | Float | ₹7,000 – ₹339,700 | Gross monthly revenue in INR (heavy-tailed log-normal) |
| `income_variability` | Float | 0.080 – 0.750 | Coefficient of variation in monthly revenues (seasonality index; 4.7% missing) |
| `num_employees` | Integer | 0 – 4 | Non-owner paid or unpaid labor count |
| `repayment_on_time_ratio`| Float | 0.400 – 1.000 | Historical fraction of EMIs paid without grace-period breach |
| `num_delayed_installments`| Integer | 0 – 9 | Total EMI installments paid post 30+ days overdue |
| `has_defaulted_past` | Binary | 0 or 1 | Past loan write-off or 90+ DPD event (overall 6.8%) |
| `mobile_banking_usage` | Categorical | None, SMS/USSD, Smartphone App/UPI | Mode of financial digital engagement |
| `digital_payment_ratio` | Float | 0.000 – 0.980 | Fraction of customer collections via UPI/QR (1.9% missing) |
| `savings_frequency` | Categorical | Never, Irregular, Monthly, Weekly | Frequency of formal/SHG deposit additions |
| `savings_balance_ratio` | Float | 0.05 – 4.00 | Total savings balance relative to average monthly revenue (4.1% missing) |
| `shg_cooperative_member`| Binary | 0 or 1 | Active membership in SHG, JLG, or Farmer Producer Org (FPO) |
| `market_access_type` | Categorical | Local Village Only, Weekly Haat, Semi-Urban Mandi, Direct Institutional | Primary sales channel breadth |
| `past_training_received`| Categorical | 0 or 1 | Prior exposure to enterprise or vocational training (3.7% missing) |
| `latent_archetype_label`| Categorical | 5 Archetypes | Ground-truth generative group (masked during unsupervised clustering) |

---

## 3. Generative Archetypes & Structural Assumptions

| Archetype | Population Share | Dominant Sector | Mean Loan (₹) | Mean Rev (₹) | On-Time Repayment | Digital Ratio | Key Feature |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stable Subsistence Retail** | 28.0% | Kirana / Retail / Textiles | ₹85,000 | ₹32,000 | 0.86 | 0.31 | Mature vintage, steady cashflow, localized market |
| **Women SHG Collective Artisans** | 21.5% | Tailoring, Crafts, Food | ₹45,000 | ₹21,000 | 0.93 | 0.22 | 96% SHG members, 94% women, high repayment discipline |
| **Agri-Allied Commercial Scalers** | 20.6% | Agri-Allied, Dairy, Processing| ₹220,000 | ₹68,000 | 0.91 | 0.61 | Multiple employees, semi-urban mandi links, high capital |
| **Vulnerable Seasonal Producers** | 17.9% | Agri-Allied, Dairy, Crafts | ₹55,000 | ₹18,000 | 0.62 | 0.11 | High cash volatility (CV > 0.45), lean-period default risk |
| **Digital-First Youth Services** | 12.0% | Services & Repairs, Retail | ₹140,000 | ₹42,000 | 0.77 | 0.76 | Young (mean age 27), smartphone-native, high leverage |

### Realistic Noise, Missingness, and Outlier Assumptions:
1. **Realistic Non-Response (MCAR/MAR):** Survey items like `income_variability` and `savings_balance_ratio` carry 2–5% missing rates, mirroring field MIS collection omissions.
2. **Right-Skewed Pareto/Log-Normal Distribution:** Revenues and loan amounts incorporate a 1.5% extreme outlier cohort reflecting local rural market monopolies.
3. **Imperfect Boundaries:** Archetypes are generated using wide Beta and Gaussian variances, ensuring clusters overlap naturally rather than behaving like artificial synthetic blobs.

> [!WARNING]
> **Synthetic Structure Recovery vs. Real-World Borrower Accuracy:**
> The high Adjusted Rand Index ($\text{ARI} = 0.982$) between K-Means clusters and `latent_archetype_label` demonstrates that the unsupervised pipeline successfully recovers the underlying 5-mixture generative structure. However, this is an **internal algorithmic sanity check on synthetic data**, not a demonstration of accuracy on real-world rural borrowers. Prospective testing on independent empirical datasets (detailed below) is mandatory prior to institutional deployment.

---

## 4. Real Public Data Equivalents (Plug-and-Play Sources)

If deploying in production or replacing synthetic data with empirical Indian databases:
1. **NABARD All India Rural Financial Inclusion Survey (NAFIS):** Detailed rural household income, debt, institutional borrowing, and livestock/crop micro-enterprise modules.
2. **MUDRA Annual Reports & PMMY MIS:** Loan tranche breakdown (Shishu: <₹50k, Kishore: ₹50k–₹5L, Tarun: ₹5L–₹10L) with default rates across commercial banks, RRBs, and MFIs.
3. **RBI Priority Sector Lending (PSL) & Financial Inclusion Index (FI-Index):** District-level credit penetration, credit-deposit ratios, and sub-target performance metrics.
4. **World Bank Global Findex Database (India Microdata):** Empirical survey of 3,000+ Indian adults tracking formal bank accounts, mobile phone transactions, emergency savings, and borrowing reasons.
5. **NSSO 77th Round (Schedule 18.2 — Debt and Investment):** Asset holdings, debt outstanding, incidence of indebtedness, and interest rates for rural operational holdings.
