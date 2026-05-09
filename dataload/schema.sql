-- =============================================================
-- Underwriting Copilot — schema for app_vllm.py
-- =============================================================
-- Run with:
--   ysqlsh -h <yb-host> -p 5433 -U yugabyte -d yugabyte -f schema.sql
-- Or:
--   psql ... -f schema.sql
--
-- Idempotent: drops + recreates the 5 demo tables. Does NOT touch
-- existing cag_corpus / rag_chunks tables.
-- =============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- -------------------------------------------------------------
-- 1. customers — master record (KYC, risk, compliance status)
-- -------------------------------------------------------------
DROP TABLE IF EXISTS decision_log CASCADE;
DROP TABLE IF EXISTS rag_files CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS cag_state CASCADE;
DROP TABLE IF EXISTS cag_policy CASCADE;

CREATE TABLE customers (
    id                  BIGSERIAL PRIMARY KEY,
    customer_code       TEXT UNIQUE NOT NULL,
    full_name           TEXT NOT NULL,
    pan_masked          TEXT,
    current_risk_grade  TEXT NOT NULL DEFAULT 'standard',
        -- 'standard' | 'watchlist' | 'npa'
    compliance_status   TEXT NOT NULL DEFAULT 'cleared',
        -- 'cleared' | 'under_review' | 'blocked'
    kyc_last_updated    DATE,
    residential_city    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX customers_compliance_idx ON customers(compliance_status);
CREATE INDEX customers_risk_idx ON customers(current_risk_grade);

-- -------------------------------------------------------------
-- 2. rag_files — past underwriting decisions with embeddings
-- -------------------------------------------------------------
CREATE TABLE rag_files (
    id                      BIGSERIAL PRIMARY KEY,
    file_number             TEXT UNIQUE NOT NULL,
    customer_id             BIGINT REFERENCES customers(id) ON DELETE SET NULL,
    decision                TEXT NOT NULL,
        -- 'sanctioned' | 'sanctioned_with_conditions' | 'rejected' | 'deferred'
    loan_type               TEXT NOT NULL,
        -- 'home_loan' | 'LAP' | 'plot_loan' | 'home_construction'
    property_city           TEXT,
    city_tier               TEXT,
        -- 'metro' | 'tier_1' | 'tier_2' | 'tier_3'
    ltv                     NUMERIC(5,2),
    foir                    NUMERIC(5,2),
    cibil_score             INT,
    property_value_lakhs    NUMERIC(10,2),
    loan_amount_lakhs       NUMERIC(10,2),
    employment_type         TEXT,
    monthly_income          NUMERIC(12,2),
    decision_date           DATE,
    summary                 TEXT,
    rationale               TEXT,
    rejection_reasons       JSONB,
    psl_eligible            BOOLEAN DEFAULT false,
    embedding               VECTOR(768),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX rag_files_decision_idx ON rag_files(decision);
CREATE INDEX rag_files_city_idx ON rag_files(property_city);
CREATE INDEX rag_files_psl_idx ON rag_files(psl_eligible) WHERE psl_eligible = true;
CREATE INDEX rag_files_customer_idx ON rag_files(customer_id);

-- HNSW vector index for fast similarity search
-- (YugabyteDB uses ybhnsw; PostgreSQL pgvector uses hnsw)
-- The loader script picks the right one based on extension version
-- For now leave the index creation to the loader to handle either case.

-- -------------------------------------------------------------
-- 3. cag_policy — versioned policy corpus (the cacheable prefix)
-- -------------------------------------------------------------
CREATE TABLE cag_policy (
    id              BIGSERIAL PRIMARY KEY,
    section_key     TEXT NOT NULL,
    title           TEXT,
    content         TEXT NOT NULL,
    source          TEXT,
        -- 'rbi_master_direction' | 'internal' | 'rbi_fema_master_direction' | etc
    version         INT NOT NULL DEFAULT 1,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    approved_by     TEXT NOT NULL DEFAULT 'system',
    approved_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (section_key, version)
);

CREATE INDEX cag_policy_active_idx ON cag_policy(is_active, section_key)
    WHERE is_active = true;

-- -------------------------------------------------------------
-- 4. cag_state — single-row tracker for the current "warm" prefix
-- -------------------------------------------------------------
CREATE TABLE cag_state (
    id              INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    prefix_version  INT NOT NULL,
    prefix_hash     TEXT NOT NULL,
    warmed_at       TIMESTAMPTZ
);

INSERT INTO cag_state (id, prefix_version, prefix_hash, warmed_at)
VALUES (1, 1, 'pending', NULL)
ON CONFLICT (id) DO NOTHING;

-- -------------------------------------------------------------
-- 5. decision_log — every assistant call writes one row here
-- -------------------------------------------------------------
CREATE TABLE decision_log (
    id              BIGSERIAL PRIMARY KEY,
    underwriter_id  TEXT NOT NULL,
    file_number     TEXT,
    query           TEXT NOT NULL,
    retrieved_ids   BIGINT[] NOT NULL,
    prefix_version  INT NOT NULL,
    response        TEXT NOT NULL,
    cached_tokens   INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX decision_log_underwriter_idx ON decision_log(underwriter_id, created_at DESC);
CREATE INDEX decision_log_created_idx ON decision_log(created_at DESC);

-- -------------------------------------------------------------
-- Sanity check: list what was created
-- -------------------------------------------------------------
SELECT 'created: ' || tablename
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('customers', 'rag_files', 'cag_policy', 'cag_state', 'decision_log')
ORDER BY tablename;
