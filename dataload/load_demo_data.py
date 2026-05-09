"""
load_demo_data.py — fills the underwriting copilot tables with synthetic
Indian home loan data.

Prereqs:
    1. schema.sql already run (creates the 5 tables)
    2. pip install psycopg2-binary pgvector sentence-transformers numpy

Usage:
    export YB_HOST=10.31.16.10
    export YB_PORT=5433
    export YB_USER=yugabyte
    export YB_PASSWORD=yugabyte
    export YB_DB=yugabyte
    python load_demo_data.py

Idempotent: TRUNCATEs the tables before loading.
Takes ~3 minutes (most of that is downloading + embedding 100 precedents).
"""
import json
import os
import random
import hashlib
from datetime import date, timedelta

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

random.seed(42)  # deterministic data

DB_CONFIG = {
    "host": os.environ.get("YB_HOST", "localhost"),
    "port": int(os.environ.get("YB_PORT", 5433)),
    "dbname": os.environ.get("YB_DB", "yugabyte"),
    "user": os.environ.get("YB_USER", "yugabyte"),
    "password": os.environ.get("YB_PASSWORD", "yugabyte"),
}

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim, matches schema
N_CUSTOMERS = 80
N_PRECEDENTS = 100


# =========================================================
# 1. POLICY CORPUS — 10 sections (~12K tokens total)
# =========================================================
POLICY_SECTIONS = [
    {
        "section_key": "system.role",
        "title": "Underwriting Assistant Role",
        "source": "internal",
        "content": (
            "You are an underwriting assistant for an Indian home loan lender "
            "regulated by the Reserve Bank of India. Your job is to help "
            "underwriters review applications by citing applicable policy and "
            "comparing the current case with relevant past decisions. "
            "Always cite policy sections by their [section_key] tag and "
            "precedent files by their file_number. End with a recommendation "
            "that the underwriter will independently review. You do not "
            "approve or reject loans on your own."
        ),
    },
    {
        "section_key": "credit_policy.ltv_caps",
        "title": "Loan-to-Value Caps (RBI Master Direction)",
        "source": "rbi_master_direction",
        "content": (
            "Per RBI Master Direction on Housing Finance (last updated 2024), "
            "loan-to-value (LTV) ratio caps for housing loans are tiered by "
            "loan amount: up to Rs 30 lakhs, LTV may not exceed 90 percent of "
            "the property value; from Rs 30 lakhs to Rs 75 lakhs, LTV may not "
            "exceed 80 percent; above Rs 75 lakhs, LTV may not exceed 75 "
            "percent. Stamp duty and registration charges shall not be "
            "included in the cost of property for LTV computation, except for "
            "loans up to Rs 10 lakhs where these may be included subject to "
            "additional documentation. For loans against property (LAP), the "
            "maximum LTV is 75 percent regardless of loan amount."
        ),
    },
    {
        "section_key": "credit_policy.foir_thresholds",
        "title": "FOIR (Fixed Obligation to Income Ratio) Thresholds",
        "source": "internal",
        "content": (
            "Fixed Obligation to Income Ratio (FOIR) measures the borrower's "
            "total monthly EMI obligations as a percentage of net monthly "
            "income. Thresholds: salaried home loans, maximum FOIR 55 percent "
            "(up to 60 percent permitted with strong compensating factors "
            "such as CIBIL above 800, monthly income above Rs 3 lakhs, or "
            "additional collateral). Self-employed home loans, maximum FOIR "
            "50 percent based on average of last 3 years ITR. Loan against "
            "property (LAP) for any segment, maximum FOIR 45 percent. "
            "Existing EMIs across all lenders must be considered, including "
            "credit card minimum payments computed at 5 percent of "
            "outstanding balance. Where FOIR exceeds the threshold by more "
            "than 5 percentage points, the file shall be referred to the "
            "Senior Credit Committee."
        ),
    },
    {
        "section_key": "credit_policy.cibil_requirements",
        "title": "CIBIL Score Requirements",
        "source": "internal",
        "content": (
            "Minimum CIBIL score thresholds: standard home loans, 700 and "
            "above; affordable housing under PMAY-CLSS, 650 and above; loan "
            "against property, 720 and above; takeover/balance transfer "
            "loans, 750 and above. Applicants with no CIBIL history (commonly "
            "called 'credit unhistoried' or first-time borrowers) may be "
            "considered subject to alternative data: rental payment history, "
            "utility bill payment history, salary credit consistency for at "
            "least 24 months. Negative remarks such as written-off accounts, "
            "settlement, or DPD over 90 days in last 12 months trigger "
            "automatic decline unless explained by documented hardship and "
            "approved by the Senior Credit Committee."
        ),
    },
    {
        "section_key": "credit_policy.compensating_factors",
        "title": "Compensating Factors for Marginal Cases",
        "source": "internal",
        "content": (
            "Where one or more credit parameters fall just outside policy "
            "thresholds, the underwriter may consider compensating factors "
            "before declining: (a) high CIBIL score above 800 with at least "
            "5 years of history; (b) liquid net worth (FDs, mutual funds, "
            "equity holdings) at least 25 percent of loan amount; (c) "
            "additional co-applicant with independent income and acceptable "
            "credit; (d) salary credit through bank for at least 36 months "
            "with employer of repute; (e) existing customer with at least 24 "
            "months of disciplined repayment on prior loans with us; (f) "
            "additional collateral (residential or commercial) with clear "
            "title, providing extra security cover. Marginal cases shall be "
            "documented in the credit memo with explicit notes on which "
            "compensating factors were applied."
        ),
    },
    {
        "section_key": "credit_policy.property_title",
        "title": "Property and Title Requirements",
        "source": "internal",
        "content": (
            "All financed properties must have clear and marketable title "
            "verified by the empanelled legal counsel. Title chain must be "
            "verifiable for at least 30 years (12 years where state law "
            "permits). The property must be registered with the appropriate "
            "sub-registrar and have an Encumbrance Certificate showing no "
            "subsisting encumbrances at the time of disbursement. CERSAI "
            "registration is mandatory for all loans above Rs 5 lakhs. RERA "
            "registration of the project is required for under-construction "
            "properties; we do not finance projects without RERA "
            "registration except for plotted land and self-construction on "
            "owned land. Litigation, partial paper title, or unauthorised "
            "construction triggers automatic decline or deferral pending "
            "regularisation."
        ),
    },
    {
        "section_key": "credit_policy.income_documentation",
        "title": "Income Documentation Requirements",
        "source": "internal",
        "content": (
            "Salaried applicants must provide: latest 3 months salary slips, "
            "Form 16 for the last 2 years, bank statements showing salary "
            "credits for the last 6 months, and employer verification "
            "letter. Self-employed applicants must provide: ITRs for the "
            "last 3 financial years with computation of income and "
            "balance-sheet, business proof (GST registration, Shop Act "
            "license, or partnership deed), bank statements for both "
            "personal and business accounts for the last 12 months, and CA "
            "certificate for income above Rs 50 lakhs. NRI applicants must "
            "additionally provide: valid passport with visa, employment "
            "contract or business proof from country of residence, and FEMA-"
            "compliant funds source declaration."
        ),
    },
    {
        "section_key": "regulatory.fair_practices_code",
        "title": "Fair Practices Code (RBI)",
        "source": "rbi_master_direction",
        "content": (
            "Under the RBI Fair Practices Code for Lenders, all loan "
            "applications shall be acknowledged within 7 working days. "
            "Decisions on completed applications shall be communicated within "
            "30 days. Rejection letters shall state specific reasons in clear "
            "language and shall be issued in the borrower's preferred "
            "vernacular language where practicable. The lender shall not "
            "discriminate on the basis of sex, caste, religion, or minority "
            "status. All terms and conditions, including processing fees, "
            "interest computation method, and prepayment charges, shall be "
            "disclosed in the sanction letter. Recovery practices shall be "
            "non-coercive and during reasonable hours, typically 7am to 7pm. "
            "Customer grievances shall be acknowledged within 10 days and "
            "resolved within 30 days, failing which the customer may "
            "approach the Banking Ombudsman."
        ),
    },
    {
        "section_key": "regulatory.psl_affordable_housing",
        "title": "Priority Sector Lending — Affordable Housing",
        "source": "rbi_master_direction",
        "content": (
            "Per RBI Master Direction on Priority Sector Lending, housing "
            "loans up to Rs 35 lakhs in metropolitan centres (population 10 "
            "lakhs and above) and up to Rs 25 lakhs in other centres, where "
            "the cost of dwelling unit does not exceed Rs 45 lakhs and Rs 30 "
            "lakhs respectively, qualify as priority sector. Loans for "
            "individuals up to Rs 15 lakhs for repair of damaged dwelling "
            "units also qualify. PMAY-CLSS interest subsidy is available for "
            "EWS, LIG, MIG-1, and MIG-2 categories subject to income and "
            "carpet-area ceilings notified by Ministry of Housing and Urban "
            "Affairs. Applicants seeking PMAY-CLSS must not have any pucca "
            "house owned by any family member anywhere in India. Aadhaar "
            "linkage of all family members is mandatory."
        ),
    },
    {
        "section_key": "regulatory.nri_fema",
        "title": "NRI Lending — FEMA Compliance",
        "source": "rbi_fema_master_direction",
        "content": (
            "Loans to Non-Resident Indians (NRIs) and Persons of Indian "
            "Origin (PIOs) for acquisition of residential property in India "
            "are governed by FEMA Master Direction on Acquisition and "
            "Transfer of Immovable Property in India. NRIs may purchase "
            "residential and commercial property in India (excluding "
            "agricultural land, plantation property, and farmhouses). Loan "
            "shall be in INR, repayment must come from NRE/NRO/FCNR account "
            "or inward remittance through normal banking channels. Original "
            "documents (passport, visa, employment proof) must be verified. "
            "TDS at applicable rate on rental income from financed property "
            "is the borrower's responsibility. Repatriation of sale proceeds "
            "is permitted up to USD 1 million per financial year subject to "
            "documentary evidence of source. All transactions must comply "
            "with the prevailing Foreign Exchange Management (Acquisition "
            "and Transfer of Immovable Property) Regulations."
        ),
    },
]


# =========================================================
# 2. PRECEDENT FILES — 100 synthetic Indian underwriting decisions
# =========================================================
INDIAN_NAMES = [
    ("Aarav", "Sharma"), ("Vivaan", "Verma"), ("Aditya", "Patel"),
    ("Vihaan", "Singh"), ("Arjun", "Kumar"), ("Sai", "Reddy"),
    ("Reyansh", "Gupta"), ("Ayaan", "Iyer"), ("Krishna", "Nair"),
    ("Ishaan", "Mehta"), ("Saanvi", "Shah"), ("Aanya", "Rao"),
    ("Aadhya", "Pillai"), ("Diya", "Joshi"), ("Pari", "Desai"),
    ("Anaya", "Bhatt"), ("Kiara", "Menon"), ("Myra", "Chopra"),
    ("Sara", "Banerjee"), ("Ahana", "Kapoor"), ("Rohan", "Malhotra"),
    ("Ritika", "Agarwal"), ("Karan", "Mishra"), ("Priya", "Pandey"),
    ("Neha", "Sinha"), ("Arnav", "Trivedi"), ("Aryan", "Choudhury"),
    ("Sneha", "Saxena"), ("Tara", "Khanna"), ("Riya", "Bhatia"),
]
CITIES = {
    "metro":   ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad"],
    "tier_1":  ["Pune", "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Kochi", "Indore"],
    "tier_2":  ["Coimbatore", "Vadodara", "Nagpur", "Visakhapatnam", "Chandigarh", "Bhubaneswar"],
    "tier_3":  ["Mysuru", "Hubli", "Madurai", "Thiruvananthapuram", "Guwahati", "Ranchi"],
}
EMPLOYMENT_TYPES = [
    "salaried_mnc", "salaried_psu", "salaried_indian_corp",
    "self_employed_professional", "self_employed_business", "nri",
]
LOAN_TYPES = ["home_loan", "LAP", "plot_loan", "home_construction"]


def gen_pan_masked() -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    p = "".join(random.choices(letters, k=5)) + \
        "".join(random.choices(digits, k=4)) + \
        random.choice(letters)
    return p[:2] + "XXXX" + p[6:]


def gen_customer(idx: int) -> dict:
    fn, ln = random.choice(INDIAN_NAMES)
    tier = random.choices(
        list(CITIES.keys()),
        weights=[40, 30, 20, 10],
        k=1
    )[0]
    city = random.choice(CITIES[tier])
    risk = random.choices(
        ["standard", "watchlist", "npa"],
        weights=[88, 10, 2],
        k=1
    )[0]
    compliance = "blocked" if risk == "npa" else random.choices(
        ["cleared", "under_review"],
        weights=[95, 5],
        k=1
    )[0]
    return {
        "customer_code": f"CUST-{idx:06d}",
        "full_name": f"{fn} {ln}",
        "pan_masked": gen_pan_masked(),
        "current_risk_grade": risk,
        "compliance_status": compliance,
        "kyc_last_updated": date(2024, random.randint(1, 12),
                                  random.randint(1, 28)),
        "residential_city": city,
    }


def gen_precedent(idx: int, customer_id: int, customer_city: str) -> dict:
    # Choose scenario template for variety
    scenario = random.choices(
        ["clean_approval", "high_ltv_approval", "marginal_foir",
         "low_cibil_reject", "title_defect_defer", "nri_fema",
         "psl_affordable", "lap_overleveraged", "self_employed_complex",
         "high_value_metro"],
        weights=[20, 12, 12, 10, 8, 8, 10, 7, 8, 5],
        k=1,
    )[0]

    loan_type = random.choices(LOAN_TYPES, weights=[66, 26, 5, 3], k=1)[0]
    tier = random.choices(["metro", "tier_1", "tier_2", "tier_3"],
                          weights=[37, 33, 19, 11], k=1)[0]
    city = random.choice(CITIES[tier])

    if scenario == "clean_approval":
        ltv = round(random.uniform(65, 78), 2)
        foir = round(random.uniform(35, 48), 2)
        cibil = random.randint(750, 820)
        decision = "sanctioned"
        rationale = (
            f"All parameters within policy. LTV {ltv}% (per "
            f"[credit_policy.ltv_caps]), FOIR {foir}% (per "
            f"[credit_policy.foir_thresholds]), CIBIL {cibil} (per "
            f"[credit_policy.cibil_requirements]). Clear title. "
            f"Standard approval."
        )
    elif scenario == "high_ltv_approval":
        ltv = round(random.uniform(80, 88), 2)
        foir = round(random.uniform(38, 50), 2)
        cibil = random.randint(770, 820)
        decision = "sanctioned_with_conditions"
        rationale = (
            f"LTV {ltv}% near upper bound for tier. Approved with conditions: "
            f"insurance assignment, higher EMI bounce charge, quarterly "
            f"property valuation. Compensating factors: CIBIL {cibil}, "
            f"strong income stability ([credit_policy.compensating_factors])."
        )
    elif scenario == "marginal_foir":
        ltv = round(random.uniform(70, 80), 2)
        foir = round(random.uniform(54, 59), 2)
        cibil = random.randint(720, 800)
        decision = "sanctioned_with_conditions" if cibil > 760 else "deferred"
        rationale = (
            f"FOIR {foir}% exceeds standard threshold of 55% per "
            f"[credit_policy.foir_thresholds]. "
            + ("Approved with conditions due to compensating factors per "
               "[credit_policy.compensating_factors]: high CIBIL "
               f"{cibil}, salary credit history."
               if decision == "sanctioned_with_conditions"
               else "Deferred pending evidence of additional income or "
                    "co-applicant.")
        )
    elif scenario == "low_cibil_reject":
        ltv = round(random.uniform(70, 82), 2)
        foir = round(random.uniform(40, 52), 2)
        cibil = random.randint(550, 690)
        decision = "rejected"
        rationale = (
            f"CIBIL score {cibil} below minimum threshold of 700 per "
            f"[credit_policy.cibil_requirements]. No documented hardship to "
            f"justify Senior Credit Committee referral. Recommend reapply "
            f"after 6 months of disciplined credit behavior."
        )
    elif scenario == "title_defect_defer":
        ltv = round(random.uniform(70, 80), 2)
        foir = round(random.uniform(42, 54), 2)
        cibil = random.randint(720, 800)
        decision = "deferred"
        rationale = (
            f"Borrower credit profile acceptable. Property title shows "
            f"partial paper, missing CERSAI registration, or unauthorised "
            f"construction. Per [credit_policy.property_title], deferred "
            f"pending regularisation."
        )
    elif scenario == "nri_fema":
        ltv = round(random.uniform(65, 75), 2)
        foir = round(random.uniform(35, 48), 2)
        cibil = random.randint(720, 810)
        decision = random.choices(
            ["sanctioned", "sanctioned_with_conditions", "deferred"],
            weights=[60, 30, 10], k=1)[0]
        rationale = (
            f"NRI applicant. FEMA compliance per [regulatory.nri_fema] "
            f"verified: passport, visa, employment proof, FCNR funds "
            f"source. " + (
                "Approved." if decision == "sanctioned"
                else "Approved with conditions on rental TDS and repatriation."
                if decision == "sanctioned_with_conditions"
                else "Deferred pending additional FEMA documentation."
            )
        )
    elif scenario == "psl_affordable":
        # PSL affordable requires lower loan and lower property value
        ltv = round(random.uniform(75, 88), 2)
        foir = round(random.uniform(38, 52), 2)
        cibil = random.randint(670, 780)
        decision = "sanctioned"
        rationale = (
            f"Affordable housing PMAY-CLSS application per "
            f"[regulatory.psl_affordable_housing]. Property cost and loan "
            f"size within priority sector limits. Aadhaar linkage verified. "
            f"Subsidy claim filed with NHB."
        )
    elif scenario == "lap_overleveraged":
        ltv = round(random.uniform(72, 78), 2)
        foir = round(random.uniform(46, 55), 2)
        cibil = random.randint(700, 770)
        decision = "rejected"
        rationale = (
            f"Loan against property request. FOIR {foir}% exceeds LAP "
            f"maximum of 45% per [credit_policy.foir_thresholds]. Existing "
            f"obligations across lenders too high. Recommend consolidation "
            f"of existing debt before fresh LAP application."
        )
    elif scenario == "self_employed_complex":
        ltv = round(random.uniform(65, 78), 2)
        foir = round(random.uniform(40, 52), 2)
        cibil = random.randint(720, 810)
        decision = random.choices(
            ["sanctioned", "sanctioned_with_conditions"],
            weights=[55, 45], k=1)[0]
        rationale = (
            f"Self-employed applicant. ITR for 3 years averaged per "
            f"[credit_policy.income_documentation]. Income volatility "
            f"observed in latest year. " + (
                "Approved on stable 3-year average."
                if decision == "sanctioned"
                else "Approved with conditions: higher processing fee, "
                     "personal guarantee from co-applicant."
            )
        )
    else:  # high_value_metro
        ltv = round(random.uniform(70, 76), 2)  # max 75% per policy
        foir = round(random.uniform(36, 48), 2)
        cibil = random.randint(770, 820)
        decision = "sanctioned"
        rationale = (
            f"High-value metro purchase. LTV capped at {ltv}% per upper "
            f"tier rule in [credit_policy.ltv_caps] (loans above Rs 75L). "
            f"Strong borrower profile, clean credit, salary above Rs 5L "
            f"per month."
        )

    # Property value / loan amount derivations
    if scenario == "high_value_metro":
        property_value = random.uniform(120, 350)  # lakhs
    elif scenario == "psl_affordable":
        property_value = random.uniform(20, 45)
    elif loan_type == "LAP":
        property_value = random.uniform(50, 200)
    else:
        property_value = random.uniform(30, 120)

    loan_amount = round(property_value * ltv / 100, 2)
    property_value = round(property_value, 2)

    employment = random.choice(EMPLOYMENT_TYPES) if scenario != "nri_fema" else "nri"
    monthly_income = round(loan_amount * 100000 * 0.012 / (foir / 100), 0)

    psl_eligible = (
        scenario == "psl_affordable" or
        (loan_amount <= 35 and tier == "metro" and property_value <= 45) or
        (loan_amount <= 25 and tier != "metro" and property_value <= 30)
    )

    decision_dt = date(2024, random.randint(1, 12), random.randint(1, 28))

    rejection_reasons = None
    if decision == "rejected":
        if scenario == "low_cibil_reject":
            rejection_reasons = ["cibil_below_threshold", "no_hardship_documented"]
        elif scenario == "lap_overleveraged":
            rejection_reasons = ["foir_exceeded", "existing_obligations_high"]
        else:
            rejection_reasons = ["policy_threshold_breach"]

    summary = (
        f"{decision.replace('_', ' ').title()} — {loan_type} for "
        f"{employment.replace('_', ' ')} applicant in {city} ({tier}). "
        f"Property Rs {property_value} lakhs, loan Rs {loan_amount} lakhs."
    )

    return {
        "file_number": f"HL-2024-{idx:06d}",
        "customer_id": customer_id,
        "decision": decision,
        "loan_type": loan_type,
        "property_city": city,
        "city_tier": tier,
        "ltv": ltv,
        "foir": foir,
        "cibil_score": cibil,
        "property_value_lakhs": property_value,
        "loan_amount_lakhs": loan_amount,
        "employment_type": employment,
        "monthly_income": monthly_income,
        "decision_date": decision_dt,
        "summary": summary,
        "rationale": rationale,
        "rejection_reasons": json.dumps(rejection_reasons) if rejection_reasons else None,
        "psl_eligible": psl_eligible,
    }


# =========================================================
# 3. LOAD INTO YUGABYTEDB
# =========================================================
def main():
    print("Connecting to YugabyteDB...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # ----- Clear existing demo data -----
    print("Clearing existing demo data (if any)...")
    cur.execute("TRUNCATE decision_log CASCADE;")
    cur.execute("TRUNCATE rag_files CASCADE;")
    cur.execute("TRUNCATE customers CASCADE;")
    cur.execute("TRUNCATE cag_policy CASCADE;")
    conn.commit()

    # ----- Load policy corpus -----
    print(f"Loading {len(POLICY_SECTIONS)} policy sections...")
    for s in POLICY_SECTIONS:
        cur.execute(
            """INSERT INTO cag_policy
               (section_key, title, content, source, version, is_active,
                approved_by, approved_at)
               VALUES (%(section_key)s, %(title)s, %(content)s, %(source)s,
                       1, true, 'compliance_officer', now())""",
            s,
        )
    conn.commit()

    # Update cag_state with policy hash
    policy_concat = "\n".join(s["section_key"] + s["content"] for s in
                              sorted(POLICY_SECTIONS, key=lambda x: x["section_key"]))
    policy_hash = hashlib.sha256(policy_concat.encode()).hexdigest()[:16]
    cur.execute(
        """INSERT INTO cag_state (id, prefix_version, prefix_hash, warmed_at)
           VALUES (1, 1, %s, now())
           ON CONFLICT (id) DO UPDATE
           SET prefix_version = EXCLUDED.prefix_version,
               prefix_hash = EXCLUDED.prefix_hash,
               warmed_at = EXCLUDED.warmed_at""",
        (policy_hash,),
    )
    conn.commit()

    # ----- Load customers -----
    print(f"Generating {N_CUSTOMERS} customer records...")
    customer_ids = []
    for i in range(1, N_CUSTOMERS + 1):
        c = gen_customer(i)
        cur.execute(
            """INSERT INTO customers
               (customer_code, full_name, pan_masked, current_risk_grade,
                compliance_status, kyc_last_updated, residential_city)
               VALUES (%(customer_code)s, %(full_name)s, %(pan_masked)s,
                       %(current_risk_grade)s, %(compliance_status)s,
                       %(kyc_last_updated)s, %(residential_city)s)
               RETURNING id""",
            c,
        )
        customer_ids.append((cur.fetchone()[0], c["residential_city"]))
    conn.commit()
    print(f"  ...{N_CUSTOMERS} customers loaded.")

    # ----- Generate precedent files -----
    print(f"Generating {N_PRECEDENTS} precedent files...")
    precedents = []
    for i in range(1, N_PRECEDENTS + 1):
        cust_id, cust_city = random.choice(customer_ids)
        p = gen_precedent(i, cust_id, cust_city)
        precedents.append(p)

    # ----- Embed all summaries -----
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    print("(First run downloads ~500 MB. Takes a minute or two.)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding precedent summaries...")
    texts = [p["summary"] + " " + p["rationale"] for p in precedents]
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=16)

    # ----- Insert precedents with embeddings -----
    print("Inserting precedents...")
    for p, emb in zip(precedents, embeddings):
        cur.execute(
            """INSERT INTO rag_files
               (file_number, customer_id, decision, loan_type, property_city,
                city_tier, ltv, foir, cibil_score, property_value_lakhs,
                loan_amount_lakhs, employment_type, monthly_income,
                decision_date, summary, rationale, rejection_reasons,
                psl_eligible, embedding)
               VALUES (%(file_number)s, %(customer_id)s, %(decision)s,
                       %(loan_type)s, %(property_city)s, %(city_tier)s,
                       %(ltv)s, %(foir)s, %(cibil_score)s,
                       %(property_value_lakhs)s, %(loan_amount_lakhs)s,
                       %(employment_type)s, %(monthly_income)s,
                       %(decision_date)s, %(summary)s, %(rationale)s,
                       %(rejection_reasons)s::jsonb, %(psl_eligible)s,
                       %(embedding)s)""",
            {**p, "embedding": emb.tolist()},
        )
    conn.commit()
    print(f"  ...{N_PRECEDENTS} precedents loaded with embeddings.")

    # ----- Build vector index -----
    print("Building HNSW vector index...")
    # YugabyteDB uses ybhnsw; PostgreSQL pgvector uses hnsw. Try both.
    try:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS rag_files_embedding_hnsw
            ON rag_files
            USING ybhnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 128)
        """)
        conn.commit()
        print("  ybhnsw index created (YugabyteDB).")
    except Exception as e1:
        conn.rollback()
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS rag_files_embedding_hnsw
                ON rag_files
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 128)
            """)
            conn.commit()
            print("  hnsw index created (pgvector).")
        except Exception as e2:
            conn.rollback()
            print(f"  WARN: HNSW index not created. ybhnsw err: {e1}")
            print(f"  WARN: hnsw err: {e2}")
            print("  App will still work but vector search will be slower (sequential scan).")

    # ----- Sanity checks -----
    print()
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    for query, label in [
        ("SELECT count(*) FROM customers", "customers"),
        ("SELECT count(*) FROM rag_files", "rag_files"),
        ("SELECT count(*) FROM cag_policy WHERE is_active = true",
         "active policy sections"),
        ("SELECT count(*) FROM cag_state", "cag_state rows"),
        ("SELECT count(*) FROM rag_files WHERE psl_eligible = true",
         "PSL-eligible files"),
    ]:
        cur.execute(query)
        n = cur.fetchone()[0]
        print(f"  {label:35s} : {n}")

    # Decision distribution
    cur.execute(
        "SELECT decision, count(*) FROM rag_files GROUP BY decision "
        "ORDER BY count(*) DESC"
    )
    print()
    print("  Decision distribution:")
    for decision, n in cur.fetchall():
        print(f"    {decision:30s} : {n}")

    # Sample blog query: vector + JOIN + filters
    print()
    print("Running blog example query (vector + customer JOIN + filter)...")
    test_query = ("Salaried metro applicant with FOIR around 50% and CIBIL "
                  "above 750, recent decision")
    test_emb = model.encode(test_query, normalize_embeddings=True).tolist()
    cur.execute(
        """SELECT rf.file_number, rf.decision, rf.property_city,
                  rf.ltv, rf.foir, rf.cibil_score, c.full_name,
                  c.compliance_status,
                  1 - (rf.embedding <=> %s::vector) AS similarity
           FROM rag_files rf
           LEFT JOIN customers c ON c.id = rf.customer_id
           WHERE c.compliance_status = 'cleared'
             AND rf.decision_date >= CURRENT_DATE - INTERVAL '2 years'
           ORDER BY rf.embedding <=> %s::vector
           LIMIT 5""",
        (test_emb, test_emb),
    )
    print()
    print("  Top 5 results:")
    for row in cur.fetchall():
        print(f"    {row[0]} | {row[1]:30s} | {row[2]:15s} | "
              f"LTV {row[3]}% | sim {row[8]:.3f}")

    cur.close()
    conn.close()
    print()
    print("=" * 60)
    print("DONE. Schema and data loaded successfully.")
    print("=" * 60)
    print()
    print("Next: run the Streamlit app:")
    print("    export VLLM_URL=http://localhost:30000")
    print("    export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct")
    print("    export YB_HOST=" + DB_CONFIG["host"])
    print("    streamlit run app_vllm.py")


if __name__ == "__main__":
    main()
