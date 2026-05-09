# Setting up the YugabyteDB schema for app_vllm.py

You have two files:

1. `schema.sql` — creates the 5 tables app_vllm.py needs
2. `load_demo_data.py` — fills them with synthetic Indian underwriting data

Run them in this order. Takes about 5 minutes total.

## Step 1: Create the schema (30 seconds)

On any machine that can reach your YugabyteDB:

```bash
# Either ysqlsh (if YB tools installed)
ysqlsh -h <yb-host> -p 5433 -U yugabyte -d yugabyte -f schema.sql

# Or psql (PostgreSQL client)
PGPASSWORD=yugabyte psql -h <yb-host> -p 5433 -U yugabyte -d yugabyte -f schema.sql
```

You should see at the end:

```
created: cag_policy
created: cag_state
created: customers
created: decision_log
created: rag_files
```

If you see `ERROR: extension "vector" is not available`, your YugabyteDB version is too old. Need YB 2.21+. Check with `SELECT version();`

## Step 2: Load the data (~5 minutes)

```bash
# Set DB connection
export YB_HOST=<yb-host>
export YB_PORT=5433
export YB_USER=yugabyte
export YB_PASSWORD=yugabyte
export YB_DB=yugabyte

# Install Python deps if not already in venv
pip install psycopg2-binary pgvector sentence-transformers numpy

# Run the loader
python load_demo_data.py
```

What happens:

1. TRUNCATEs the 5 tables (safe — they're freshly created and empty anyway)
2. Inserts 10 policy sections into `cag_policy` (~12K tokens of RBI + internal policy)
3. Generates 80 synthetic customers (~88% with `compliance_status='cleared'`)
4. Generates 100 precedent files across 10 realistic Indian underwriting scenarios
5. Embeds all 100 precedents using BGE-base (768-dim)
   - First run downloads the model (~500 MB) to `~/.cache/huggingface`
   - Subsequent runs reuse the cached model
6. Builds an HNSW vector index for fast similarity search
7. Runs sanity checks and prints a sample blog query result

Expected output at the end:

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

Running blog example query (vector + customer JOIN + filter)...

  Top 5 results:
    HL-2024-000023 | sanctioned                     | Bengaluru       | LTV 75.4% | sim 0.812
    HL-2024-000089 | sanctioned_with_conditions     | Mumbai          | LTV 78.0% | sim 0.804
    HL-2024-000041 | sanctioned                     | Delhi           | LTV 72.1% | sim 0.787
    HL-2024-000067 | sanctioned                     | Chennai         | LTV 73.5% | sim 0.776
    HL-2024-000012 | sanctioned_with_conditions     | Hyderabad       | LTV 76.8% | sim 0.770

============================================================
DONE. Schema and data loaded successfully.
============================================================
```

## Step 3: Run the Streamlit app

```bash
export VLLM_URL=http://localhost:30000
export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct
export YB_HOST=<yb-host>
streamlit run app_vllm.py --server.fileWatcherType none --server.address 0.0.0.0 --server.port 8501
```

The `--server.fileWatcherType none` flag silences those torchvision warnings from earlier (Streamlit's source watcher poking at every transformers model). It's purely cosmetic; the app works either way.

## Where to run the loader

You have a few options. Easiest first:

**Option A: Run on the GPU VM** (where you already have python + a venv)
```bash
# In your existing vllm-env or streamlit-env
pip install psycopg2-binary pgvector sentence-transformers
python load_demo_data.py
```

**Option B: Run on the AlmaLinux YB box** (zero network latency)
- Most efficient because embeddings happen co-located with the database
- Requires you to set up Python and pgvector there

**Option C: Run on the bastion** (cleanest separation)
- Bastion already has network access to YB
- Doesn't bloat the GPU VM with sentence-transformers

Option A is fine for a one-off demo.

## Common errors

**`ERROR:  extension "vector" is not available`**
Your YugabyteDB version is too old. Need YB 2.21+. The pgvector extension must be available in the cluster's package list.

**`psycopg2.OperationalError: could not connect to server`**
Network/firewall issue. Verify YB_HOST is reachable: `nc -zv <yb-host> 5433`

**`OOM during sentence-transformers download`**
Model download is ~500 MB. If your VM has < 4 GB RAM free, embedding may OOM. Try a smaller embedding model — but then update both `schema.sql` (change `VECTOR(768)` to match) and the loader (change `EMBEDDING_MODEL`).

**`ERROR: relation "cag_corpus" does not exist`** (after dropping)
You're not seeing this — but if you ever query `cag_corpus` and it's gone, that's because we don't touch it. Your old `cag_corpus` and `rag_chunks` tables are untouched by `schema.sql`.

## What this gives you

After running both files, your YugabyteDB has:

| Table | Rows | Purpose |
|-------|------|---------|
| `customers` | 80 | KYC + risk grade + compliance status |
| `rag_files` | 100 | Past precedent decisions with embeddings |
| `cag_policy` | 10 | Policy corpus (cacheable prefix source) |
| `cag_state` | 1 | Current warm prefix version + hash |
| `decision_log` | 0 (fills as Streamlit runs) | Audit trail |

Plus your existing `cag_corpus` and `rag_chunks` are untouched — they remain alongside.

The Streamlit app will use the new tables (`cag_policy`, `rag_files`, `customers`, `decision_log`) and ignore the old ones.
