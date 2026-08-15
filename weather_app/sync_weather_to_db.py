"""Sync weather data from NWS API to Lakebase Postgres database.

Orchestrates fetching weather documents and upserting them into the database.
Generates and stores embeddings for document text.
"""

import logging
from typing import Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2.extras import execute_values

from .weather_client import WeatherFetcher
from . import lakebase

logger = logging.getLogger(__name__)


def _ensure_table_exists():
    """Create the weather_documents table if it doesn't exist."""
    # First ensure the schema exists
    lakebase.run_query('CREATE SCHEMA IF NOT EXISTS "hw-2"')
    
    create_table_sql = """
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
        
        -- Index for location queries
        CREATE INDEX IF NOT EXISTS idx_weather_location ON "hw-2".weather_documents(location);
        
        -- Index for time-based queries
        CREATE INDEX IF NOT EXISTS idx_weather_issued ON "hw-2".weather_documents(issued_at);
        
        -- Index for source type filtering
        CREATE INDEX IF NOT EXISTS idx_weather_source ON "hw-2".weather_documents(source_type);
    """
    
    lakebase.run_query(create_table_sql)
    logger.info("Ensured weather_documents table exists")


def _upsert_documents(documents: list[dict]) -> int:
    """Upsert weather documents into the database.
    
    Args:
        documents: List of normalized document dicts
        
    Returns:
        Number of documents upserted
    """
    if not documents:
        return 0
    
    upsert_sql = """
        INSERT INTO "hw-2".weather_documents (
            id, location, location_lat, location_lon, source_type,
            headline, event, severity, narrative_text,
            issued_at, effective_at, expires_at, payload, synced_at
        ) VALUES (
            %(id)s, %(location)s, %(location_lat)s, %(location_lon)s, %(source_type)s,
            %(headline)s, %(event)s, %(severity)s, %(narrative_text)s,
            %(issued_at)s, %(effective_at)s, %(expires_at)s, %(payload)s, %(synced_at)s
        )
        ON CONFLICT (id) DO UPDATE SET
            location = EXCLUDED.location,
            location_lat = EXCLUDED.location_lat,
            location_lon = EXCLUDED.location_lon,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            event = EXCLUDED.event,
            severity = EXCLUDED.severity,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            expires_at = EXCLUDED.expires_at,
            payload = EXCLUDED.payload,
            synced_at = EXCLUDED.synced_at
    """
    
    # Convert payload dicts to JSON strings for JSONB
    import json
    for doc in documents:
        if isinstance(doc.get('payload'), dict):
            doc['payload'] = json.dumps(doc['payload'])
    
    lakebase.run_query(upsert_sql, documents, executemany=True)
    logger.info(f"Upserted {len(documents)} documents")
    
    return len(documents)


def _ensure_embeddings_table_exists(embedding_dim: int):
    """Create the weather_embeddings table if it doesn't exist.
    
    Args:
        embedding_dim: Dimensionality of the embedding vectors
    """
    create_embeddings_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "hw-2".weather_embeddings (
            embedding_id SERIAL PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES "hw-2".weather_documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            model_name TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(document_id, chunk_index, model_name, chunk_size, chunk_overlap)
        );
        
        -- Index for vector similarity search
        CREATE INDEX IF NOT EXISTS idx_weather_embeddings_vector 
            ON "hw-2".weather_embeddings 
            USING hnsw (embedding vector_cosine_ops);
        
        -- Index for joining with documents
        CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id 
            ON "hw-2".weather_embeddings(document_id);
        
        -- Index for model and config lookups
        CREATE INDEX IF NOT EXISTS idx_weather_embeddings_config 
            ON "hw-2".weather_embeddings(model_name, chunk_size, chunk_overlap);
    """
    
    lakebase.run_query(create_embeddings_table_sql)
    logger.info("Ensured weather_embeddings table exists")


def _chunk_text_sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks using sliding window.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text or len(text) == 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move window by (chunk_size - overlap)
        start += (chunk_size - overlap)
        
        # Break if we've covered the entire text
        if end >= len(text):
            break
    
    return chunks


def _get_unembedded_document_ids(
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    document_ids: list[str] = None
) -> list[str]:
    """Get document IDs that don't have embeddings for the given config.
    
    Args:
        model_name: Embedding model name
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        document_ids: Optional list of specific document IDs to check (defaults to all)
        
    Returns:
        List of document IDs that need embeddings
    """
    if document_ids:
        # Use parameterized query with ANY() for proper SQL array handling
        query = """
            SELECT d.id
            FROM "hw-2".weather_documents d
            LEFT JOIN "hw-2".weather_embeddings e 
                ON d.id = e.document_id 
                AND e.model_name = %s
                AND e.chunk_size = %s
                AND e.chunk_overlap = %s
            WHERE e.document_id IS NULL
                AND d.id = ANY(%s)
            ORDER BY d.synced_at DESC
        """
        results = lakebase.run_query(query, (model_name, chunk_size, chunk_overlap, document_ids))
    else:
        query = """
            SELECT d.id
            FROM "hw-2".weather_documents d
            LEFT JOIN "hw-2".weather_embeddings e 
                ON d.id = e.document_id 
                AND e.model_name = %s
                AND e.chunk_size = %s
                AND e.chunk_overlap = %s
            WHERE e.document_id IS NULL
            ORDER BY d.synced_at DESC
        """
        results = lakebase.run_query(query, (model_name, chunk_size, chunk_overlap))
    
    return [row['id'] for row in results]


def _generate_embeddings_for_documents(
    document_ids: list[str],
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int = 32,
    max_workers: int = 4
) -> int:
    """Generate and store embeddings for a list of documents.
    
    Args:
        document_ids: List of document IDs to embed
        model_name: Name of the embedding model
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        batch_size: Batch size for embedding generation
        max_workers: Max parallel workers for embedding
        
    Returns:
        Number of chunks embedded
    """
    if not document_ids:
        logger.info("No documents to embed")
        return 0
    
    # Import SentenceTransformer here to avoid import at module level
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Skipping embedding generation.")
        return 0
    
    # Load the embedding model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Fetch documents using parameterized query
    query = """
        SELECT 
            id,
            location,
            narrative_text,
            event
        FROM "hw-2".weather_documents
        WHERE id = ANY(%s)
    """
    
    documents = lakebase.run_query(query, (document_ids,))
    logger.info(f"Fetched {len(documents)} documents for embedding")
    
    # Prepare all chunks
    all_chunks = []
    for doc in documents:
        # Concatenate fields (handle None values)
        parts = [
            doc.get('location') or '',
            doc.get('narrative_text') or '',
            doc.get('event') or ''
        ]
        full_text = ' '.join(p for p in parts if p).strip()
        
        # Chunk the text
        chunks = _chunk_text_sliding_window(full_text, chunk_size, chunk_overlap)
        
        # Store chunk info
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                'document_id': doc['id'],
                'chunk_index': chunk_idx,
                'chunk_text': chunk_text
            })
    
    if not all_chunks:
        logger.info("No text chunks to embed")
        return 0
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    logger.info(f"Average chunks per document: {len(all_chunks) / len(documents):.1f}")
    
    # Generate embeddings in batches and insert using execute_values
    total_embedded = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_texts = [chunk['chunk_text'] for chunk in batch]
        
        # Generate embeddings
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        # Prepare data for batch insert using execute_values
        values_list = [
            (
                chunk_info['document_id'],
                chunk_info['chunk_index'],
                chunk_info['chunk_text'],
                embedding.tolist(),  # Convert numpy array to list
                model_name,
                chunk_size,
                chunk_overlap,
                datetime.now()
            )
            for chunk_info, embedding in zip(batch, embeddings)
        ]
        
        # Use execute_values for batch insert with ON CONFLICT
        insert_sql = """
            INSERT INTO "hw-2".weather_embeddings 
                (document_id, chunk_index, chunk_text, embedding, model_name, chunk_size, chunk_overlap, created_at)
            VALUES %s
            ON CONFLICT (document_id, chunk_index, model_name, chunk_size, chunk_overlap) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                created_at = EXCLUDED.created_at
        """
        
        # Use psycopg2's execute_values for efficient batch insert
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                # Template with explicit cast to vector type
                template = "(%s, %s, %s, %s::vector, %s, %s, %s, %s)"
                execute_values(
                    cur,
                    insert_sql,
                    values_list,
                    template=template,
                    page_size=batch_size
                )
                conn.commit()
        
        total_embedded += len(batch)
        batch_num = i // batch_size + 1
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        logger.info(f"Embedded batch {batch_num}/{total_batches}: {len(batch)} chunks ({total_embedded}/{len(all_chunks)} total)")
    
    logger.info(f"Successfully embedded {total_embedded} chunks from {len(documents)} documents")
    return total_embedded


def sync_weather_data(config: dict) -> dict[str, Any]:
    """Fetch weather data and sync to database.
    
    Args:
        config: Configuration dict with keys:
            - locations: List of "City, ST" strings or (lat, lon) tuples
            - limit: Max documents to fetch PER CITY (default: 50)
            - include_alerts: Fetch alerts (default: True)
            - include_forecasts: Fetch forecasts (default: True)
            - include_hourly: Fetch hourly forecasts (default: False)
            - embeddings: Optional dict with embedding config:
                - enabled: Whether to generate embeddings (default: True)
                - model_name: Embedding model name (default: "sentence-transformers/all-MiniLM-L6-v2")
                - embedding_dim: Embedding dimensions (default: 384)
                - chunk_size: Chunk size in characters (default: 500)
                - chunk_overlap: Chunk overlap in characters (default: 100)
                - batch_size: Batch size for embedding generation (default: 32)
                - max_workers: Max parallel workers (default: 4)
            
    Returns:
        Dict with keys:
            - documents_fetched: Number of documents fetched
            - documents_upserted: Number of documents written to DB
            - chunks_embedded: Number of chunks embedded
            - stats: Fetch statistics
            - errors: List of errors encountered
    """
    logger.info("Starting weather data sync")
    
    # Ensure table exists
    _ensure_table_exists()
    
    # Fetch weather documents
    fetcher = WeatherFetcher(rate_limit_delay=0.1)
    fetch_result = fetcher.fetch_weather_documents(config)
    
    documents = fetch_result['documents']
    stats = fetch_result['stats']
    errors = fetch_result['errors']
    
    # Upsert to database
    try:
        documents_upserted = _upsert_documents(documents)
    except Exception as e:
        logger.error(f"Failed to upsert documents: {e}")
        errors.append(f"Database upsert failed: {e}")
        documents_upserted = 0
    
    # Generate embeddings if enabled
    chunks_embedded = 0
    embeddings_config = config.get('embeddings', {})
    embeddings_enabled = embeddings_config.get('enabled', True)  # Default to enabled
    
    if embeddings_enabled and documents:
        try:
            logger.info("Starting embedding generation")
            
            # Extract embedding configuration with defaults
            model_name = embeddings_config.get(
                'model_name',
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            embedding_dim = embeddings_config.get('embedding_dim', 384)
            chunk_size = embeddings_config.get('chunk_size', 500)
            chunk_overlap = embeddings_config.get('chunk_overlap', 100)
            batch_size = embeddings_config.get('batch_size', 32)
            max_workers = embeddings_config.get('max_workers', 4)
            
            # Ensure embeddings table exists
            _ensure_embeddings_table_exists(embedding_dim)
            
            # Get list of document IDs that were just upserted
            upserted_doc_ids = [doc['id'] for doc in documents]
            
            # Check which documents need embeddings
            unembedded_ids = _get_unembedded_document_ids(
                model_name,
                chunk_size,
                chunk_overlap,
                upserted_doc_ids
            )
            
            if unembedded_ids:
                logger.info(f"Found {len(unembedded_ids)} documents without embeddings")
                chunks_embedded = _generate_embeddings_for_documents(
                    unembedded_ids,
                    model_name,
                    chunk_size,
                    chunk_overlap,
                    batch_size,
                    max_workers
                )
            else:
                logger.info("All documents already have embeddings")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            errors.append(f"Embedding generation failed: {e}")
    elif not embeddings_enabled:
        logger.info("Embedding generation is disabled")
    
    logger.info(
        f"Sync complete: {len(documents)} fetched, {documents_upserted} upserted, "
        f"{chunks_embedded} chunks embedded"
    )
    
    return {
        "documents_fetched": len(documents),
        "documents_upserted": documents_upserted,
        "chunks_embedded": chunks_embedded,
        "stats": stats,
        "errors": errors,
    }
