"""
Underwriting Copilot — Streamlit UI (vLLM + few-shot examples edition)

Updated to handle BOTH home loans AND gold loans with explicit few-shot
examples that teach the model the exact output format we want.

Major changes from app_vllm.py:
  1. Detects loan domain from query keywords (gold vs home) and biases
     retrieval filters accordingly
  2. Includes 3 few-shot examples in the prompt (1 home + 2 gold)
     showing the exact citation + recommendation format expected
  3. Adds a "Loan domain" filter in the sidebar
  4. Stop tokens tuned for the structured output format

Run:
    pip install streamlit psycopg2-binary pgvector sentence-transformers requests
    export VLLM_URL=http://localhost:30000
    export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct
    export YB_HOST=<host>
    streamlit run app_vllm.py --server.fileWatcherType none \\
        --server.address 0.0.0.0 --server.port 8501
"""
import json
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import psycopg2
import requests
import streamlit as st
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("YB_HOST", "localhost"),
    "port": int(os.environ.get("YB_PORT", 5433)),
    "dbname": os.environ.get("YB_DB", "yugabyte"),
    "user": os.environ.get("YB_USER", "yugabyte"),
    "password": os.environ.get("YB_PASSWORD", "yugabyte"),
}

VLLM_URL = os.environ.get("VLLM_URL", "").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

LIVE_MODE = bool(VLLM_URL)


# ---------------------------------------------------------------
# Few-shot examples — these teach the model the output format
# ---------------------------------------------------------------
FEW_SHOT_EXAMPLES = """\
# EXAMPLE 1 — Home loan, FOIR exceeded, conditional approval

## Question
Salaried MNC employee in Pune, FOIR 56%, CIBIL 780, LTV 78%, requesting Rs 75 lakh home loan. Recommend?

## Retrieved precedent
Past file HL-2024-000023 (decision: sanctioned_with_conditions). home_loan in Pune (tier_1). LTV 76%, FOIR 56.2%, CIBIL 790. Property Rs 95L, loan Rs 72L. Rationale: FOIR exceeded standard 55% by margin; compensating factors of CIBIL above 800 and salary credit history applied.

## Analysis
The applicant's FOIR of 56% breaches the standard threshold of 55% per [credit_policy.foir_thresholds]. However, the margin is small (1 percentage point) and the applicant has a strong CIBIL score of 780 indicating disciplined credit history. Per [credit_policy.compensating_factors], a CIBIL above 800 is a recognised compensating factor; at 780 the score is just below this threshold but still strong. LTV of 78% is comfortably within the 80% cap for the Rs 30-75 lakh tier per [credit_policy.ltv_caps]. Precedent file HL-2024-000023 in the same tier-1 city was sanctioned with conditions on similar terms.

## Recommendation
SANCTIONED WITH CONDITIONS. Approve at applied LTV of 78%. Conditions: (a) auto-debit mandate for EMI, (b) salary account to be opened with the bank for credit, (c) mandatory life insurance assignment. Refer to Senior Credit Committee if any of these cannot be obtained at sanction.

# EXAMPLE 2 — Gold loan, small ticket, clean approval

## Question
Housewife in Trichy pledging 22 grams of 22-carat gold jewellery for Rs 1.2 lakh personal loan, purpose declared as medical. Recommend?

## Retrieved precedent
Past file GL-2024-000087 (decision: sanctioned). gold_loan (bullet_12m) for housewife_no_income in Trichy (tier_2). Net gold 21.5g at 22-carat, eligible value Rs 1.46 lakhs, loan Rs 1.18 lakhs, LTV 80.8%. Purpose: personal_medical.

## Analysis
This is a small-ticket gold loan within the Rs 2.5 lakh tier. The proposed LTV of approximately 80% is well within the RBI cap of 85% for this tier per [gold.regulatory.rbi_ltv_caps]. The collateral (22-carat jewellery) is fully eligible per [gold.regulatory.eligible_collateral]. Acid-and-touchstone assay is sufficient at this ticket size per [gold.regulatory.assaying_valuation], with a two-appraiser sign-off required. Simplified eKYC applies per [gold.regulatory.kyc_aml_pmla] for loans up to Rs 2.5 lakh. Disbursement must be to the borrower's bank account (NEFT/UPI) per [gold.regulatory.disbursement_cash_limit] since the amount exceeds Rs 20,000. Precedent file GL-2024-000087 sanctioned on near-identical terms.

## Recommendation
SANCTIONED. Standard 12-month bullet repayment scheme. Disbursement via NEFT to the customer's bank account on completion of pledge sealing and KYC verification. No additional conditions beyond standard.

# EXAMPLE 3 — Gold loan, ineligible collateral, rejection

## Question
Customer in Coimbatore wants Rs 4 lakh loan against 80 grams of gold bars purchased from a jeweller. Recommend?

## Retrieved precedent
Past file GL-2024-000142 (decision: rejected). gold_loan for shop_owner in Coimbatore (tier_1). Gross gold 75g but presented as bullion bars. Rationale: Primary gold (bars/bullion) explicitly NOT eligible per RBI Directions 2025.

## Analysis
The proposed collateral is 80 grams of gold bars purchased as an investment product. Per [gold.regulatory.eligible_collateral], primary gold including bars, bullion, ingots, and gold biscuits is EXPLICITLY NOT eligible as collateral under the RBI Directions on Lending Against Gold and Silver Collateral, 2025. This is a hard regulatory bar; it is not subject to compensating factors or override. The only eligible forms are (a) gold jewellery and ornaments of purity not less than 18 carat, or (b) specially-minted gold coins issued by banks (not jewellers) of purity not less than 22 carat, with a per-borrower aggregate cap of 50 grams. Precedent file GL-2024-000142 was rejected on identical grounds.

## Recommendation
REJECTED. Provide the customer with a written rejection letter in their preferred vernacular language per [gold.regulatory.fair_practices_code], stating the regulatory ground for rejection. Counsel the customer that the bank cannot lend against bars but may consider a fresh application if the customer is able to pledge eligible jewellery or specially-minted bank coins (within the 50g aggregate limit).
"""

OUTPUT_FORMAT_INSTRUCTION = """\
You will receive (a) the policy reference above, (b) some retrieved precedent files, and (c) a question. Answer in the same format as the examples:

## Analysis
[Cite specific policy section_keys in square brackets like [gold.regulatory.rbi_ltv_caps]. Cite specific precedent file_numbers like HL-2024-000023. Walk through what the policy says about this case and which precedents are most analogous. Keep this to 4-7 sentences.]

## Recommendation
[One of: SANCTIONED, SANCTIONED WITH CONDITIONS, DEFERRED, or REJECTED. Then a short paragraph stating exactly what the underwriter should do next, including any conditions or required documentation.]

Be specific. Cite policy section_keys and precedent file_numbers. Do not invent facts not present in the policy or precedents. If the case truly does not match any policy section, say so and recommend referral to the Credit Risk team.
"""


# ---------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------
def get_vllm_metrics() -> dict:
    if not VLLM_URL:
        return {}
    try:
        resp = requests.get(f"{VLLM_URL}/metrics", timeout=2)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return {}
    queries = 0.0
    hits = 0.0
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("vllm:prefix_cache_queries_total") or \
           line.startswith("vllm:gpu_prefix_cache_queries_total"):
            try: queries = float(line.split()[-1])
            except (ValueError, IndexError): pass
        elif line.startswith("vllm:prefix_cache_hits_total") or \
             line.startswith("vllm:gpu_prefix_cache_hits_total"):
            try: hits = float(line.split()[-1])
            except (ValueError, IndexError): pass
    hit_rate = (hits / queries * 100.0) if queries > 0 else 0.0
    return {
        "cache_queries_total": queries,
        "cache_hits_total": hits,
        "cache_hit_rate": hit_rate,
    }


def call_vllm(policy_prefix, precedents, query, underwriter_id,
              file_number, max_tokens=600):
    if not VLLM_URL:
        raise RuntimeError("VLLM_URL not set")
    metrics_before = get_vllm_metrics()

    precedent_block = "\n\n".join([
        f"### Past file {p['file_number']} (decision: {p['decision']})\n"
        f"{p['loan_type']} in {p['property_city']} ({p['city_tier']}). "
        f"LTV {p['ltv']}%, "
        + (f"FOIR {p['foir']}%, " if p['foir'] and p['foir'] > 0 else "")
        + (f"CIBIL {p['cibil_score']}, " if p['cibil_score'] and p['cibil_score'] > 0 else "")
        + f"Property/collateral Rs {p['property_value_lakhs']} lakhs, "
        f"loan Rs {p['loan_amount_lakhs']} lakhs.\n"
        f"Rationale: {p['rationale']}"
        for p in precedents
    ])

    # Construct the full prompt:
    # [POLICY PREFIX (cacheable, identical across requests)]
    # [FEW-SHOT EXAMPLES (cacheable, identical across requests)]
    # [OUTPUT FORMAT INSTRUCTION (cacheable)]
    # [PRECEDENTS BLOCK (varies)]
    # [QUESTION (varies)]
    full_prompt = (
        f"{policy_prefix}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}\n\n"
        f"# CURRENT REQUEST\n\n"
        f"## Retrieved precedents\n\n{precedent_block}\n\n"
        f"## Question\n"
        f"Underwriter: {underwriter_id}\n"
        f"File: {file_number or 'not specified'}\n"
        f"{query}\n\n"
        f"## Analysis\n"
    )

    t0 = time.perf_counter()
    resp = requests.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model": VLLM_MODEL,
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["\n# EXAMPLE", "\n## Question\n", "\nUnderwriter:"],
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000

    metrics_after = get_vllm_metrics()
    queries_delta = metrics_after.get("cache_queries_total", 0) - \
                    metrics_before.get("cache_queries_total", 0)
    hits_delta = metrics_after.get("cache_hits_total", 0) - \
                 metrics_before.get("cache_hits_total", 0)
    call_hit_rate = (hits_delta / queries_delta * 100.0) if queries_delta > 0 else 0.0

    text = data["choices"][0]["text"].strip()
    # Re-prepend the "## Analysis" header that the prompt cut off
    if not text.startswith("##"):
        text = "## Analysis\n" + text
    usage = data.get("usage", {})

    meta = {
        "latency_ms": latency_ms,
        "model": VLLM_MODEL,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "this_call_cache_queries": int(queries_delta),
        "this_call_cache_hits": int(hits_delta),
        "this_call_hit_rate": call_hit_rate,
        "cumulative_cache_queries": int(metrics_after.get("cache_queries_total", 0)),
        "cumulative_cache_hits": int(metrics_after.get("cache_hits_total", 0)),
        "cumulative_hit_rate": metrics_after.get("cache_hit_rate", 0),
        "is_cache_hit": hits_delta > 0,
    }
    return text, meta


# ---------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------
def detect_domain(query: str) -> str:
    """Detect loan domain from the query keywords."""
    q = query.lower()
    gold_keywords = ["gold", "jewellery", "jewelry", "ornament", "carat",
                     "ltv on gold", "pledge", "bullion", "kdm", "916",
                     "auction", "appraiser", "assay"]
    home_keywords = ["home loan", "house loan", "property", "flat",
                     "apartment", "foir", "cibil", "ltv on home",
                     "title", "psl home", "pmay", "rera"]
    gold_score = sum(1 for kw in gold_keywords if kw in q)
    home_score = sum(1 for kw in home_keywords if kw in q)
    if gold_score > home_score:
        return "gold"
    if home_score > gold_score:
        return "home"
    return "any"  # ambiguous


# ---------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


@st.cache_data(ttl=300)
def load_policy_prefix(domain_filter: str = "all") -> tuple[str, int]:
    """Load policy prefix. domain_filter: 'all', 'gold', 'home'."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if domain_filter == "gold":
                cur.execute("""
                    SELECT section_key, content, source, version
                    FROM cag_policy
                    WHERE is_active = TRUE
                      AND section_key LIKE 'gold.%'
                    ORDER BY section_key
                """)
            elif domain_filter == "home":
                cur.execute("""
                    SELECT section_key, content, source, version
                    FROM cag_policy
                    WHERE is_active = TRUE
                      AND section_key NOT LIKE 'gold.%'
                    ORDER BY section_key
                """)
            else:  # all
                cur.execute("""
                    SELECT section_key, content, source, version
                    FROM cag_policy
                    WHERE is_active = TRUE
                    ORDER BY section_key
                """)
            rows = cur.fetchall()
            if not rows:
                return ("", 0)
            version = max(r[3] for r in rows)
            parts = [
                f"## [{section_key}] (source: {source}, v{ver})\n{content}"
                for section_key, content, source, ver in rows
            ]
            prefix = (
                "# Indian Lending Policy Reference\n\n"
                "You are an underwriting assistant. Use ONLY the policies below "
                "and the precedent files provided to answer questions. Always "
                "cite policy sections by their [section_key] tag and precedents "
                "by file_number.\n\n"
                + "\n\n".join(parts)
            )
            return (prefix, version)
    finally:
        conn.close()


def search_precedents(query_text, k=5, city_filter=None,
                      decision_filter=None, loan_type_filter=None):
    embedder = get_embedder()
    embedding = embedder.encode(query_text, normalize_embeddings=True)

    conn = get_conn()
    t0 = time.perf_counter()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT
                    rf.id, rf.file_number, rf.decision,
                    rf.loan_type, rf.property_city, rf.city_tier,
                    rf.ltv, rf.foir, rf.cibil_score,
                    rf.property_value_lakhs, rf.loan_amount_lakhs,
                    rf.summary, rf.rationale, rf.psl_eligible,
                    rf.employment_type,
                    c.full_name AS customer_name,
                    c.current_risk_grade AS risk_grade,
                    c.compliance_status,
                    1 - (rf.embedding <=> %s::vector) AS similarity
                FROM rag_files rf
                LEFT JOIN customers c ON c.id = rf.customer_id
                WHERE 1=1
            """
            params = [embedding.tolist()]
            if city_filter:
                sql += " AND rf.property_city = %s"
                params.append(city_filter)
            if decision_filter:
                sql += " AND rf.decision = %s"
                params.append(decision_filter)
            if loan_type_filter:
                sql += " AND rf.loan_type = %s"
                params.append(loan_type_filter)
            sql += " ORDER BY rf.embedding <=> %s::vector LIMIT %s"
            params.extend([embedding.tolist(), k])
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

    return rows, (time.perf_counter() - t0) * 1000


def log_decision(underwriter_id, file_number, query, retrieved_ids,
                 prefix_version, response, cached_tokens):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO decision_log
                    (underwriter_id, file_number, query, retrieved_ids,
                     prefix_version, response, cached_tokens, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id
            """, (underwriter_id, file_number, query, retrieved_ids,
                  prefix_version, response, cached_tokens))
            log_id = cur.fetchone()[0]
            conn.commit()
            return log_id
    finally:
        conn.close()


def mock_response(query, precedents):
    if "gold" in query.lower() or "jewellery" in query.lower():
        analysis = ("This appears to be a gold loan query. Per "
                    "[gold.regulatory.rbi_ltv_caps], LTV is tiered: 85% up to "
                    "Rs 2.5L, 80% up to Rs 5L, 75% above. Recommend reviewing "
                    "purity assay and KYC level required for the ticket size.")
    else:
        analysis = ("This appears to be a home loan query. Per "
                    "[credit_policy.ltv_caps] and [credit_policy.foir_thresholds], "
                    "the relevant tiered thresholds apply.")
    cited = ", ".join(p["file_number"] for p in precedents[:3])
    return (
        f"## Analysis\n{analysis}\n\nSimilar precedents to review: {cited}.\n\n"
        f"## Recommendation\nMock mode active. Set `VLLM_URL` to enable real "
        f"inference with prefix caching."
    ), {"latency_ms": 50, "mock": True}


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Underwriting Copilot", page_icon="🏦",
                   layout="wide")

if "underwriter_id" not in st.session_state:
    st.session_state["underwriter_id"] = f"UW-{uuid.uuid4().hex[:8].upper()}"
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

with st.sidebar:
    st.title("🏦 Underwriting Copilot")
    st.caption("RAG + CAG via YugabyteDB + vLLM")
    st.divider()

    st.subheader("LLM Backend")
    if LIVE_MODE:
        try:
            r = requests.get(f"{VLLM_URL}/v1/models", timeout=3)
            if r.status_code == 200:
                st.success("🟢 vLLM connected")
                st.caption(f"`{VLLM_URL}`")
                st.caption(f"Model: `{VLLM_MODEL}`")
            else:
                st.warning(f"🟡 vLLM HTTP {r.status_code}")
        except Exception as e:
            st.error(f"🔴 Cannot reach vLLM: {type(e).__name__}")
    else:
        st.warning("🟡 Mock mode")
        st.caption("Set `VLLM_URL` to enable.")

    if LIVE_MODE:
        m = get_vllm_metrics()
        if m:
            st.divider()
            st.subheader("Server cache (cumulative)")
            st.metric("Total queries", f"{int(m.get('cache_queries_total', 0)):,}")
            st.metric("Total hits",
                      f"{int(m.get('cache_hits_total', 0)):,}",
                      f"{m.get('cache_hit_rate', 0):.1f}% hit rate")

    st.divider()
    st.subheader("Underwriter")
    st.text_input("ID", key="underwriter_id")
    st.text_input("File # (optional)", key="file_number",
                  placeholder="HL-2025-XXXXXX or GL-2025-XXXXXX")

    st.divider()
    st.subheader("Filters")
    domain = st.selectbox(
        "Loan domain",
        options=["auto-detect", "gold loans only", "home loans only", "all"],
        index=0,
        help="auto-detect chooses based on query keywords; "
             "or force-select to see only one domain.",
    )
    city_filter = st.selectbox(
        "City filter (optional)",
        options=["", "Bengaluru", "Mumbai", "Chennai", "Delhi", "Hyderabad",
                 "Pune", "Coimbatore", "Madurai", "Kochi", "Trichy",
                 "Mysuru", "Salem", "Vadodara", "Surat", "Lucknow"],
        index=0,
    ) or None
    decision_filter = st.selectbox(
        "Decision filter (optional)",
        options=["", "sanctioned", "sanctioned_with_conditions",
                 "rejected", "deferred"],
        index=0,
    ) or None
    k_precedents = st.slider("Precedents to retrieve", 3, 10, 5)


st.title("Underwriting Decision Support")
st.caption(f"Active underwriter: {st.session_state['underwriter_id']}")

# Example questions for both domains
home_examples = [
    "Salaried MNC employee in Bengaluru, FOIR 52%, CIBIL 760, LTV 78% — recommend?",
    "NRI from Singapore wants Rs 1.5 Cr loan for property in Mumbai. FEMA compliance?",
    "Builder property in tier-2 city, missing CERSAI registration — defer or reject?",
]
gold_examples = [
    "Housewife in Madurai pledging 25g 22-carat jewellery for Rs 1.4 lakh medical loan. Recommend?",
    "Smallholder farmer in Mysuru pledges 40g gold for Rs 2.3 lakh agriculture loan. PSL?",
    "Customer in Chennai wants Rs 6 lakh against 100g of 22-carat jewellery. Process?",
    "Customer presented gold bars purchased from a jeweller. Can we accept?",
    "Trader in Coimbatore with 5-year vintage wants top-up of Rs 50K on existing pledge. OK?",
    "Walk-in customer wants full Rs 3 lakh disbursement in cash. Allowed?",
]

with st.expander("💡 Example questions — home loans", expanded=False):
    for q in home_examples:
        if st.button(q, key=f"hex_{hash(q)}", use_container_width=True):
            st.session_state["pending_query"] = q
            st.rerun()

with st.expander("💡 Example questions — gold loans", expanded=True):
    for q in gold_examples:
        if st.button(q, key=f"gex_{hash(q)}", use_container_width=True):
            st.session_state["pending_query"] = q
            st.rerun()

default_q = st.session_state.pop("pending_query", "")
query = st.text_area("Underwriter question", value=default_q, height=100)
go = st.button("🔍 Analyse", type="primary", disabled=not query.strip())


def run_analysis(query):
    # Detect or use forced domain
    if domain == "gold loans only":
        active_domain, loan_type_filter = "gold", "gold_loan"
    elif domain == "home loans only":
        active_domain, loan_type_filter = "home", None
        # Could also filter by loan_type IN ('home_loan','LAP','plot_loan',...)
    elif domain == "all":
        active_domain, loan_type_filter = "all", None
    else:  # auto-detect
        d = detect_domain(query)
        if d == "gold":
            active_domain, loan_type_filter = "gold", "gold_loan"
        elif d == "home":
            active_domain, loan_type_filter = "home", None
        else:
            active_domain, loan_type_filter = "all", None

    with st.spinner(f"Loading {active_domain} policy..."):
        prefix, version = load_policy_prefix(active_domain)
    if not prefix:
        st.error(f"No active policy found for domain '{active_domain}'.")
        return

    with st.spinner(f"Retrieving top-{k_precedents} precedents..."):
        precedents, retrieval_ms = search_precedents(
            query, k=k_precedents,
            city_filter=city_filter,
            decision_filter=decision_filter,
            loan_type_filter=loan_type_filter,
        )
    if not precedents:
        st.warning("No precedents found. Try widening filters.")
        return

    with st.spinner(f"Generating analysis via vLLM..." if LIVE_MODE
                    else "Generating mock response..."):
        if LIVE_MODE:
            try:
                response_text, meta = call_vllm(
                    policy_prefix=prefix,
                    precedents=precedents,
                    query=query,
                    underwriter_id=st.session_state["underwriter_id"],
                    file_number=st.session_state.get("file_number") or None,
                )
            except Exception as e:
                st.error(f"vLLM call failed: {e}")
                response_text, meta = mock_response(query, precedents)
        else:
            response_text, meta = mock_response(query, precedents)

    retrieved_ids = [int(p["id"]) for p in precedents]
    try:
        log_id = log_decision(
            underwriter_id=st.session_state["underwriter_id"],
            file_number=st.session_state.get("file_number") or None,
            query=query,
            retrieved_ids=retrieved_ids,
            prefix_version=version,
            response=response_text,
            cached_tokens=meta.get("this_call_cache_hits"),
        )
    except Exception as e:
        st.warning(f"Audit log write failed: {e}")
        log_id = None

    st.session_state["last_result"] = {
        "query": query,
        "domain": active_domain,
        "precedents": precedents,
        "retrieval_ms": retrieval_ms,
        "response": response_text,
        "llm_meta": meta,
        "log_id": log_id,
        "prefix_version": version,
    }
    st.session_state["history"].append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "query": query[:80] + ("..." if len(query) > 80 else ""),
        "log_id": log_id,
        "domain": active_domain,
        "is_cache_hit": meta.get("is_cache_hit", False),
    })


if go:
    run_analysis(query.strip())

result = st.session_state.get("last_result")
if result:
    precedents = result["precedents"]
    retrieval_ms = result["retrieval_ms"]
    response = result["response"]
    llm_meta = result["llm_meta"]

    # Show domain badge
    st.markdown(f"**Domain:** `{result['domain']}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precedents retrieved", len(precedents))
    col2.metric("Retrieval (DB)", f"{retrieval_ms:.0f} ms")
    col3.metric("LLM latency", f"{llm_meta.get('latency_ms', 0):.0f} ms")

    if llm_meta.get("mock"):
        col4.metric("Mode", "Mock")
    else:
        hits = llm_meta.get("this_call_cache_hits", 0)
        queries = llm_meta.get("this_call_cache_queries", 0)
        if hits > 0:
            col4.metric("Cache hit", f"{hits:,} tok",
                        delta=f"{llm_meta.get('this_call_hit_rate', 0):.0f}%")
        elif queries > 0:
            col4.metric("Cache miss", f"0 / {queries:,}",
                        delta="cold call", delta_color="off")
        else:
            col4.metric("Mode", "Live")

    st.markdown("### 📋 Analysis")
    st.markdown(response)

    if not llm_meta.get("mock"):
        with st.expander("🔍 Prompt cache details (CAG metrics)"):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Hits this call",
                       f"{llm_meta.get('this_call_cache_hits', 0):,}")
            mc2.metric("Queries this call",
                       f"{llm_meta.get('this_call_cache_queries', 0):,}")
            mc3.metric("Prompt tokens",
                       f"{llm_meta.get('prompt_tokens', 0):,}")
            mc4.metric("Output tokens",
                       f"{llm_meta.get('completion_tokens', 0):,}")
            st.divider()
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Cumulative queries",
                       f"{llm_meta.get('cumulative_cache_queries', 0):,}")
            sc2.metric("Cumulative hits",
                       f"{llm_meta.get('cumulative_cache_hits', 0):,}")
            sc3.metric("Overall hit rate",
                       f"{llm_meta.get('cumulative_hit_rate', 0):.1f}%")
            if llm_meta.get('this_call_cache_hits', 0) > 0:
                st.success("✅ Cache HIT — policy prefix served from cache.")
            else:
                st.info("🔧 Cache MISS — first time this exact prefix was seen.")

    st.markdown("### 📚 Retrieved precedents")
    for i, p in enumerate(precedents, 1):
        is_gold = p['loan_type'] == 'gold_loan'
        icon = "🟡" if is_gold else "🏠"
        with st.expander(
            f"{icon} {i}. **{p['file_number']}** — {p['decision']} "
            f"(similarity {p['similarity']:.3f})"
        ):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("LTV", f"{p['ltv']}%")
            if is_gold:
                cc2.metric("Eligible value", f"₹{p['property_value_lakhs']}L")
                cc3.metric("Loan", f"₹{p['loan_amount_lakhs']}L")
            else:
                cc2.metric("FOIR", f"{p['foir']}%")
                cc3.metric("CIBIL", p['cibil_score'])
            cc4, cc5, cc6 = st.columns(3)
            if is_gold:
                cc4.metric("Borrower", p.get('employment_type', '—').replace('_', ' '))
                cc5.metric("City", p['property_city'])
                cc6.metric("Tier", p['city_tier'])
            else:
                cc4.metric("Property", f"₹{p['property_value_lakhs']}L")
                cc5.metric("Loan", f"₹{p['loan_amount_lakhs']}L")
                cc6.metric("City tier", p['city_tier'])
            st.markdown(f"**Customer:** {p.get('customer_name', '—')} "
                        f"(risk: {p.get('risk_grade', '—')}, "
                        f"compliance: {p.get('compliance_status', '—')})")
            if p.get("psl_eligible"):
                st.markdown("🎯 **PSL eligible**")
            st.markdown(f"**Summary:** {p['summary']}")
            st.markdown(f"**Rationale:** {p['rationale']}")

    if result.get("log_id"):
        st.divider()
        st.markdown("### 📜 Audit trail")
        st.success(f"Decision logged as ID **{result['log_id']}** "
                   f"(prefix v{result['prefix_version']}).")
        st.code(f"""SELECT
    dl.query, dl.response, dl.created_at,
    (SELECT jsonb_agg(jsonb_build_object('section', section_key, 'version', version))
     FROM cag_policy WHERE version = dl.prefix_version) AS policy_snapshot,
    (SELECT jsonb_agg(jsonb_build_object('file', file_number, 'decision', decision))
     FROM rag_files WHERE id = ANY(dl.retrieved_ids)) AS precedents_shown
FROM decision_log dl
WHERE dl.id = {result['log_id']};""", language="sql")

if st.session_state["history"]:
    with st.expander(f"📋 Session history ({len(st.session_state['history'])} queries)"):
        for h in reversed(st.session_state["history"][-10:]):
            cache_indicator = "✅" if h.get("is_cache_hit") else "❄️"
            domain_icon = "🟡" if h.get("domain") == "gold" else \
                          "🏠" if h.get("domain") == "home" else "🔀"
            st.markdown(
                f"`{h['ts']}` {cache_indicator} {domain_icon} "
                f"**#{h.get('log_id', '?')}** — {h['query']}"
            )
