# Underwriting Copilot - Setup Guide

A multi-product underwriting copilot for Indian lending: home loans and gold loans, both running on the same YugabyteDB schema, the same vLLM instance, and the same Streamlit UI. One architecture, two products, one audit trail.

This guide takes you from empty database to working demo in three steps.

---

## What's in this bundle


| File                      | Purpose                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema.sql`              | Creates the 5 tables (idempotent). Doesn't touch any pre-existing tables.                                                                         |
| `load_demo_data.py`       | Loads 80 home loan customers, 100 home loan precedents, and 10 home loan policy sections. Builds the HNSW vector index.                           |
| `gold_policy_corpus.json` | 12 gold loan policy sections grounded in RBI Directions 2025 + Section 269SS.                                                                     |
| `load_gold_data.py`       | Adds 150 gold customers, 200 gold precedents, and 12 gold policy sections (additive — does not delete home loan data). Bumps the prefix version. |
| `app_vllm_v2.py`          | Streamlit app with 3 few-shot examples (1 home + 2 gold) and domain auto-detection.                                                               |

After running both loaders, you end up with:


| Asset                                                | Count |
| ---------------------------------------------------- | ----- |
| Customers (home + gold)                              | 230   |
| Precedents (home + gold + LAP + plot + construction) | 300   |
| Active policy sections                               | 22    |

---

## Lender persona

The data and policy reflect a **mid-size Indian private bank** with semi-automated underwriting:

- Small-ticket loans (≤ ₹2.5L gold, ≤ ₹30L home) follow a rapid-disbursement workflow with simplified KYC
- Larger loans require detailed credit assessment and four-eye sign-off
- Policy tone is professional but operational, not bureaucratic

---

## Prerequisites

- YugabyteDB 2025.x with `pgvector` extension available
- A vLLM server running with `--enable-prefix-caching` (we use a Tesla T4 GPU and hosted in Azure Standard NC4as T4 v3 (4 vcpus, 28 GiB memory))
- Python 3.10+ with: `psycopg2-binary`, `pgvector`, `sentence-transformers`, `streamlit`, `requests`
- Network access from your loader machine to YugabyteDB on port 5433
- Network access from your Streamlit machine to vLLM on port 30000

---

## Step 1: Create the schema (30 seconds)

# Set DB connection

export YB_HOST=10.31.16.10 (Example)
export YB_PORT=5433
export YB_USER=yugabyte
export YB_PASSWORD=xxxxxxxx
export YB_DB=yugabyte

# Run the schema script

ysqlsh -h $YB_HOST -p $YB_PORT -U $YB_USER -d $YB_DB -f schema.sql

This creates five tables: `customers`, `rag_files`, `cag_policy`, `cag_state`, `decision_log`. The script is idempotent — drops and recreates the demo tables, leaves any pre-existing tables alone.

Verify:

```sql
\dt
-- expect: cag_policy, cag_state, customers, decision_log, rag_files
```

---

## Step 2: Load home loan data (~3 minutes)

```bash
# Same env vars as above

# Install Python deps
pip install psycopg2-binary pgvector sentence-transformers numpy

# Run the home loan loader
python load_demo_data.py
```

What gets loaded:

- 10 home loan policy sections (LTV caps, FOIR thresholds, CIBIL requirements, Fair Practices Code, PSL, NRI FEMA, etc.)
- 80 customers (88% cleared, 10% watchlist, 2% NPA)
- 100 precedents across realistic scenarios (clean approval, marginal FOIR, low CIBIL reject, NRI FEMA, PSL affordable, LAP overleveraged)
- All 100 precedents embedded with `BAAI/bge-base-en-v1.5` (768-dim)
- HNSW vector index built (tries `ybhnsw` first for YugabyteDB, falls back to standard `hnsw`)

Expected sanity-check output:

```
============================================================
SANITY CHECKS
============================================================
  customers                           : 80
  rag_files                           : 100
  active policy sections              : 10
  cag_state rows                      : 1
  PSL-eligible files                  : 14

  Decision distribution:
    sanctioned                     : 49
    sanctioned_with_conditions     : 26
    rejected                       : 17
    deferred                       :  8
============================================================
DONE.
============================================================
```

---

## Step 3: Load gold loan data (~2-3 minutes)

```bash
# Same env vars

# Make sure both gold files are in the same directory
ls gold_policy_corpus.json load_gold_data.py

# Run the gold loan loader (additive)
python load_gold_data.py
```

The embedding model is cached from Step 2, so this won't re-download.

What gets loaded:

- 12 gold loan policy sections covering RBI Directions 2025 (tiered LTV: 85% / 80% / 75%), eligible collateral (jewellery yes, bars no, 50g coin cap), assaying requirements, KYC/PMLA, Section 269SS cash limit, NPA/auction process, Fair Practices Code, relationship overrides
- 150 gold customers (housewives, traders, farmers, salaried — typical gold loan demographics)
- 200 precedents across 10 realistic scenarios

Expected output:

```
[1/5] Inserting 12 gold policy sections (additive)...
  Inserted 12 sections.
  cag_state updated with new prefix_hash a3f7c2e8...

[2/5] Generating 150 gold-loan customers (additive)...
  Inserted 150 gold customers.

[3/5] Generating 200 gold loan precedents...

[4/5] Embedding 200 precedent summaries...

[5/5] Sanity checks:
============================================================
  gold customers                                : 150
  gold precedents                               : 200
  active gold policy sections                   : 12
  TOTAL customers (home + gold)                 : 230
  TOTAL precedents (home + gold)                : 300
  TOTAL active policy sections                  : 22
============================================================
DONE — gold loan data loaded alongside existing home loan data.
============================================================
```

---

## Step 4: Run the Streamlit app

```bash
# vLLM connection
export VLLM_URL=http://localhost:30000 (i.e. Azure VM which has GPU to support vLLM)
export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct

# YugabyteDB connection (same as loader steps)
export YB_HOST=10.31.16.10
export YB_PORT=5433
export YB_USER=yugabyte
export YB_PASSWORD=yugabyte

# Launch
streamlit run app_vllm.py \
    --server.fileWatcherType none \
    --server.address 0.0.0.0 \
    --server.port 8501
```

The `--server.fileWatcherType none` flag silences harmless `transformers` import warnings. Open `http://<vm-ip>:8501` in your browser. The sidebar should show **🟢 vLLM connected**.

---

## What's in the UI

### Domain auto-detection

The app reads keywords in the underwriter's question and automatically routes to the right policy and precedent set:

- Mentions of jewellery, gold, ornaments, carat, pledge → **gold domain**
- Mentions of home loan, FOIR, CIBIL, property, RERA → **home domain**
- Ambiguous → searches across all 22 policy sections and all 300 precedents

You can override the auto-detection via the **Loan domain** dropdown in the sidebar:


| Setting           | Behavior                                                      |
| ----------------- | ------------------------------------------------------------- |
| `auto-detect`     | Detect from query keywords (default)                          |
| `gold loans only` | Force gold domain                                             |
| `home loans only` | Force home domain                                             |
| `all`             | Use the combined corpus (largest prompt, broadest precedents) |

### Few-shot examples in the prompt

Every prompt includes three worked examples that teach the model the exact output format:


| Example | Domain    | Decision pattern                              |
| ------- | --------- | --------------------------------------------- |
| 1       | Home loan | FOIR exceeded → sanctioned with conditions   |
| 2       | Gold loan | Small-ticket clean → sanctioned              |
| 3       | Gold loan | Ineligible collateral (gold bars) → rejected |

Each example shows the exact `## Analysis` (with `[section_key]` citations and precedent file references) and `## Recommendation` format. This dramatically improves output quality and consistency on a 3B-parameter base model.

### Sample questions

The UI has two expandable sections with click-to-fill examples:

- Home loan examples** — 3 questions covering FOIR, CIBIL, NRI/FEMA scenarios
- **Gold loan examples** — 6 questions covering small-ticket, top-up, ineligible collateral, cash limit breaches

### Domain-aware precedent display

Retrieved precedents are tagged with the appropriate icon (gold or home), and the metric cards show domain-relevant fields:


| Domain | Cards shown                                           |
| ------ | ----------------------------------------------------- |
| Gold   | LTV, eligible value, loan amount, borrower occupation |
| Home   | LTV, FOIR, CIBIL, property value, loan amount         |

---

## Why the prompt got bigger (and why that's OK)

The full prompt now has three layers:

1. **Policy prefix** (~4-5K tokens, depending on domain filter)
2. **Few-shot examples** (~1.5K tokens, identical across all calls)
3. **Per-call content** (retrieved precedents + question, ~500 tokens, varies)

Layers 1 and 2 are byte-identical across all calls within a domain. **vLLM's prefix cache hits on both** — so a longer prompt actually delivers better results without proportional latency cost. After one cold call per domain, subsequent calls within that domain hit cache for ~5-6K tokens of prefix.

You'll see this in the cache metrics expander — the `Hits this call` value will be 4,500+ on the second and subsequent calls within a domain.

---

## Test sequences to capture for the blog

### Sequence A: Sustained cache within a domain

Run these in order. First call warms cache; second hits it.

```
Query 1: "Housewife in Trichy pledging 25g 22-carat jewellery for Rs 1.4 lakh medical loan"
Query 2: "Farmer in Mysuru pledges 40g gold for Rs 2.3 lakh agriculture loan"
```

What to expect:

- **Query 1**: cache miss, ~2-3 sec latency, cumulative cache hits unchanged
- **Query 2**: cache HIT, ~0.9 sec latency, `Hits this call` ≥ 4,500 tokens

### Sequence B: Cross-domain warmup (the killer demo)

Four queries that show the cache behavior across both products:


| # | Query                                              | Domain | Expected cache state    |
| - | -------------------------------------------------- | ------ | ----------------------- |
| 1 | Housewife pledges 25g jewellery for ₹1.4L medical | gold   | Cold (miss)             |
| 2 | Trader 5-yr vintage wants ₹50K top-up             | gold   | Hit (gold prefix warm)  |
| 3 | Salaried Bengaluru, FOIR 52%, CIBIL 760            | home   | Cold (different prefix) |
| 4 | NRI Singapore, ₹1.5 Cr Mumbai, FEMA?              | home   | Hit (home prefix warm)  |

After Sequence B, vLLM holds **two warm prefixes simultaneously** — one for each domain. Sustained queries into either domain run sub-second.

---

## Verifying the multi-product audit trail

After running Sequence B, this single SQL query reconstructs the cross-product audit:

```sql
SELECT
    dl.id,
    dl.created_at::time(0) AS time,
    LEFT(dl.query, 60) AS query,
    (SELECT loan_type FROM rag_files
     WHERE id = dl.retrieved_ids[1]) AS domain,
    array_length(dl.retrieved_ids, 1) AS n_precedents,
    dl.cached_tokens,
    dl.prefix_version
FROM decision_log dl
ORDER BY dl.id DESC
LIMIT 10;
```

Expected output:

```
 id |  time    | query                                | domain    | n | cached | v
----+----------+--------------------------------------+-----------+---+--------+--
  4 | 12:34:01 | NRI from Singapore wants Rs 1.5 Cr   | home_loan | 5 |   4823 | 2
  3 | 12:33:42 | Salaried MNC employee in Bengaluru   | home_loan | 5 |      0 | 2
  2 | 12:33:18 | Trader in Coimbatore with 5-year     | gold_loan | 5 |   4901 | 2
  1 | 12:32:55 | Housewife in Madurai pledging 25g    | gold_loan | 5 |      0 | 2
```

This single table proves three things at a glance:

- Decisions across multiple products live in one audit log
- All four reference the same `prefix_version` (v2) — the same policy snapshot
- Cold calls have `cached_tokens = 0`; warm calls have `cached_tokens` ≥ 4,800

If a regulator asks *"show me how you decided home loan HL-2024-000023 and gold loan GL-2024-000087,"* one SQL query returns both, joined to the customers, joined to the policy version active at that time.

---

## Distribution of the synthetic data

### Home loan precedents (100 total)


| Scenario                         | Count | Decision pattern           |
| -------------------------------- | ----- | -------------------------- |
| Clean approval (within all caps) | ~30   | All sanctioned             |
| High LTV approval                | ~12   | Sanctioned with conditions |
| Marginal FOIR                    | ~12   | Conditional or deferred    |
| Low CIBIL reject                 | ~10   | All rejected               |
| Title defect defer               | ~8    | All deferred               |
| NRI / FEMA                       | ~8    | Mostly sanctioned          |
| PSL affordable (PMAY-CLSS)       | ~10   | All sanctioned             |
| LAP overleveraged                | ~7    | All rejected               |
| Self-employed complex            | ~8    | Sanctioned or conditional  |
| High-value metro                 | ~5    | All sanctioned             |

### Gold loan precedents (200 total)


| Scenario                                   | Count | Decision pattern                              |
| ------------------------------------------ | ----- | --------------------------------------------- |
| Small-ticket clean (≤ ₹2.5L)             | ~60   | Mostly sanctioned                             |
| Medium-ticket standard (₹2.5-5L)          | ~36   | 80% sanctioned, 20% conditional               |
| Large-ticket enhanced (> ₹5L)             | ~16   | 55% conditional, 35% sanctioned, 10% deferred |
| PSL agriculture                            | ~24   | All sanctioned                                |
| PSL micro-enterprise                       | ~16   | 70% sanctioned, 30% conditional               |
| Underkarat dispute                         | ~10   | All deferred                                  |
| Stone-set jewellery                        | ~10   | 60% conditional, 40% sanctioned               |
| Cash disbursement breach                   | ~8    | 75% conditional (account-only), 25% rejected  |
| Ineligible collateral (bars / coins > 50g) | ~10   | All rejected                                  |
| Top-up renewal                             | ~10   | All sanctioned                                |

PSL eligible across both products: realistic mix typical of a mid-size private bank's PSL classification appetite.

---

## Cost reminder

The Tesla T4 VM costs roughly ₹44/hour while running. If it's idle, set auto-shutdown:

```bash
az vm auto-shutdown \
    --resource-group YUGABYTE-RG \
    --name bseetharaman-airbyte-vm \
    --time 1900
```

When stopped: pay only for OS disk (~₹500/month). When running: ₹44/hour.

---
