# Underwriting Copilot: RAG + CAG on YugabyteDB + vLLM

A home-loan underwriting copilot for Indian lending, built on **YugabyteDB** (vector search, customer master, versioned policy, and audit log) and **vLLM** with prefix caching.

**One database. One audit trail. Swappable LLM.**

> *Models change every few months. Regulated data shouldn't.*

---

## Why this exists

Indian lending decisions must be explainable on demand. To an RBI inspector auditing the file years later. To the customer asking why in their own language. To the bank's credit committee reviewing it next quarter.

Manual underwriting is too slow to scale. AI alone is too opaque to trust. This project shows a third path: a copilot that combines **RAG** (retrieval of past decisions), **CAG** (cached policy context), and **vLLM** (self-hosted inference) on a single distributed Postgres-compatible database, **YugabyteDB**.

Every decision is one write. Every audit reconstruction is one query.

---

## Demo metrics on a real Tesla T4

Measured on a single Azure NC4as_T4_v3 VM (1× Tesla T4, 16 GB VRAM, FP16):


| Metric                            | Value     |
| --------------------------------- | --------- |
| Prefix cache hit rate (warm call) | **97 %**  |
| Latency (cold call)               | ~2,475 ms |
| Latency (warm call)               | ~934 ms   |
| Latency reduction                 | **62 %**  |
| Customers loaded                  | 80        |
| Precedents loaded                 | 100       |
| Active policy sections            | 10        |

---

## Repo layout

```
underwritting-copilot-vllm/
├── apps/
│   └── app_vllm.py              # Streamlit UI calling vLLM
├── dataload/
│   ├── schema.sql               # 5-table DDL (idempotent)
│   ├── load_demo_data.py        # Loads home loan synthetic data with embeddings
│   └── SCHEMA_README.md         # Schema reference
└── README.md                    # This file
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (port 8501)                 │
└────┬────────────────────────────────────────┬───────────────┘
     │                                        │
     │ vector search + JOIN + audit insert    │ /v1/completions
     ▼                                        ▼
┌──────────────────────────────┐    ┌──────────────────────┐
│       YugabyteDB             │    │   vLLM 0.20.1        │
│                              │    │   Tesla T4, FP16     │
│ - customers (KYC + risk)     │    │   --enable-prefix-   │
│ - rag_files (precedents +    │    │     caching          │
│   pgvector embeddings)       │    │   Qwen2.5-3B-Instruct│
│ - cag_policy (versioned)     │    │   port 30000         │
│ - cag_state (current prefix) │    │                      │
│ - decision_log (audit trail) │    └──────────────────────┘
└──────────────────────────────┘
```

**Streamlit** runs the underwriter UI and orchestrates calls to the database and the LLM.

**YugabyteDB** holds the regulated assets: customer master, vector embeddings, versioned policy, cache state, and the audit log, under one ACID-distributed Postgres.

**vLLM** runs `Qwen/Qwen2.5-3B-Instruct` at FP16 on a Tesla T4 with `--enable-prefix-caching`in Azure VM Standard NC4as T4 v3 (4 vcpus, 28 GiB memory), so the policy prefix is processed once and reused across every subsequent request.

---

## Prerequisites

- YugabyteDB 2025.x with the `pgvector` extension available
- A vLLM server running with `--enable-prefix-caching` (Tesla T4 used here, on Azure NC4as_T4_v3, 4 vCPU, 28 GiB RAM)
- Python 3.10+
- Network access from the loader machine to YugabyteDB on port 5433
- Network access from the Streamlit machine to vLLM on port 30000

---

## Quick start

### Step 1: Create the schema (~30 sec)

```bash
export YB_HOST=10.31.16.10        # your YugabyteDB host
export YB_PORT=5433
export YB_USER=yugabyte
export YB_PASSWORD=xxxxxxxx
export YB_DB=yugabyte

ysqlsh -h $YB_HOST -p $YB_PORT -U $YB_USER -d $YB_DB -f dataload/schema.sql
```

Creates five tables: `customers`, `rag_files`, `cag_policy`, `cag_state`, `decision_log`. Idempotent: re-running is safe.

See [dataload/SCHEMA_README.md](dataload/SCHEMA_README.md) for column-level details.

### Step 2: Load home loan data (~3 min)

```bash
pip install psycopg2-binary pgvector sentence-transformers numpy

cd dataload
python load_demo_data.py
```

Loads 10 home loan policy sections (LTV caps, FOIR thresholds, CIBIL requirements, Fair Practices Code, PSL, NRI FEMA), 80 customers, and 100 precedents across realistic scenarios (clean approval, marginal FOIR, low CIBIL reject, NRI FEMA, PSL affordable, LAP overleveraged). Embeds all precedents with `BAAI/bge-base-en-v1.5` (768-dim) and builds an HNSW vector index.

Expected output:

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
```

### Step 3: Run the Streamlit app

```bash
export VLLM_URL=http://<gpu-vm-ip>:30000
export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct
export YB_HOST=10.31.16.10
export YB_PORT=5433
export YB_USER=yugabyte
export YB_PASSWORD=xxxxxxxx

cd apps
streamlit run app_vllm.py \
    --server.fileWatcherType none \
    --server.address 0.0.0.0 \
    --server.port 8501
```

Open `http://<vm-ip>:8501` in your browser. The sidebar shows **🟢 vLLM connected**.

---

## Test sequence: sustained prefix cache

Run two queries one after another. The first warms the cache; the second hits it.

```
Query 1: "Salaried MNC employee in Bengaluru, FOIR 52%, CIBIL 760, LTV 78%. Recommend?"
Query 2: "NRI from Singapore wants Rs 1.5 Cr loan for property in Mumbai. FEMA compliance?"
```

Expected behaviour:

- **Query 1**: cache miss, ~2.5 sec latency
- **Query 2**: cache HIT, ~0.9 sec latency, `Hits this call` ≥ 2,400 tokens

The latency drop and the cache-hit-token count are visible in the UI's "Prefix cache details" expander after each call.

---

## Audit reconstruction

This single SQL reconstructs the audit trail:

```sql
SELECT
    dl.id,
    dl.created_at::time(0) AS time,
    LEFT(dl.query, 60) AS query,
    array_length(dl.retrieved_ids, 1) AS n_precedents,
    dl.cached_tokens,
    dl.prefix_version
FROM decision_log dl
ORDER BY dl.id DESC
LIMIT 10;
```

For full reconstruction (question, response, policy snapshot, precedents shown):

```sql
SELECT
    dl.query, dl.response, dl.created_at,
    (SELECT jsonb_agg(jsonb_build_object(
        'section', section_key, 'version', version, 'content', content))
     FROM cag_policy WHERE version = dl.prefix_version) AS policy_snapshot,
    (SELECT jsonb_agg(jsonb_build_object(
        'file', file_number, 'decision', decision, 'rationale', rationale))
     FROM rag_files WHERE id = ANY(dl.retrieved_ids)) AS precedents_shown
FROM decision_log dl
WHERE dl.id = $1;
```

One query returns the full state of what the underwriter and the model saw at decision time.

---

## Synthetic data distribution

100 home loan precedents across realistic underwriting scenarios:


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

---

## Why YugabyteDB

Most AI architectures split data across a vector database, a relational database, and a separate audit store. The result: three systems to keep in sync, three places where consistency can drift, and three teams to coordinate during an audit.

YugabyteDB collapses all of it into one distributed SQL layer: vectors, customer records, policy, and audit trail under a single ACID transaction. The architectural pattern is engine-agnostic on the LLM side. vLLM, SGLang, Anthropic API, Ollama, or TGI all work behind the same Streamlit UI.


## Cost note

The Tesla T4 VM on Azure costs roughly ₹44/hour while running. Set auto-shutdown when idle:

```bash
az vm auto-shutdown \
    --resource-group YUGABYTE-RG \
    --name <your-vm-name> \
    --time 1900
```

Stopped VM costs only the OS disk (~₹500/month).

---

## Roadmap

Adding a new product is a data-only exercise. New policy sections, new precedents tagged with a new `loan_type`, no schema change. Planned additions:

- **Gold loan support**: RBI Directions 2025, 200 synthetic precedents
- **Silver loan support**: RBI Directions 2025 cover both gold and silver
- **LTV monitoring batch**: daily job to recompute LTV on active gold loans against IBJA rates
- **Vernacular response generation**: Fair Practices Code rejection letters in the borrower's preferred language

---

## License

MIT
