-- ============================================================================
-- Lakebase Postgres Schema Setup for HW-2
-- ============================================================================
-- This script creates the hw-2 schema and all required tables for the
-- weather data pipeline, including embeddings support.
--
-- Usage:
--   Execute this script once on your Lakebase Postgres endpoint after
--   enabling it to set up the schema and tables.
-- ============================================================================

-- Create the hw-2 schema
CREATE SCHEMA IF NOT EXISTS "hw-2";

-- ============================================================================
-- Table: weather_documents
-- Purpose: Store weather data from National Weather Service API
-- ============================================================================

CREATE TABLE IF NOT EXISTS "hw-2".weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    location_lat DOUBLE PRECISION,
    location_lon DOUBLE PRECISION,
    source_type TEXT NOT NULL,
    headline TEXT,
    event TEXT,
    severity TEXT,
    narrative_text TEXT,
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    payload JSONB,
    synced_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT check_source_type CHECK (source_type IN ('alert', 'forecast', 'hourly_forecast'))
);

-- Index for time-based queries (kept for date-only queries)
CREATE INDEX IF NOT EXISTS idx_weather_issued 
    ON "hw-2".weather_documents(issued_at);

-- Composite index for dashboard queries (source_type + issued_at filters)
CREATE INDEX IF NOT EXISTS idx_weather_source_issued 
    ON "hw-2".weather_documents(source_type, issued_at);

-- Composite index for city detail queries (location + source_type + issued_at filters)
CREATE INDEX IF NOT EXISTS idx_weather_location_source_issued 
    ON "hw-2".weather_documents(location, source_type, issued_at);

-- ============================================================================
-- Table: weather_embeddings
-- Purpose: Store vector embeddings for weather document text chunks
-- Note: Replace {EMBEDDING_DIM} with your model's embedding dimension
--       (e.g., 384 for all-MiniLM-L6-v2, 768 for sentence-transformers/all-mpnet-base-v2)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- Example for all-MiniLM-L6-v2 (384 dimensions):
CREATE TABLE IF NOT EXISTS "hw-2".weather_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES "hw-2".weather_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,  -- Change 384 to your model's dimension
    model_name TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index, model_name, chunk_size, chunk_overlap)
);

-- Index for vector similarity search (HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_vector 
    ON "hw-2".weather_embeddings 
    USING hnsw (embedding vector_cosine_ops);

-- Index for joining with documents
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id 
    ON "hw-2".weather_embeddings(document_id);

-- Index for model and config lookups
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_config 
    ON "hw-2".weather_embeddings(model_name, chunk_size, chunk_overlap);

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Check that tables were created successfully:
-- SELECT table_name 
-- FROM information_schema.tables 
-- WHERE table_schema = 'hw-2' 
-- ORDER BY table_name;

-- Check indexes:
-- SELECT indexname, tablename 
-- FROM pg_indexes 
-- WHERE schemaname = 'hw-2' 
-- ORDER BY tablename, indexname;
