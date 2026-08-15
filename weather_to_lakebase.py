# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Weather to Lakebase Pipeline
# MAGIC %md
# MAGIC # Weather to Lakebase Pipeline
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Fetches weather data from the National Weather Service API for **major US cities** (defined in weather_client)
# MAGIC 2. Connects to Lakebase Postgres using Databricks secrets
# MAGIC 3. Uses the `sync_weather_to_db` module to orchestrate data sync
# MAGIC 4. Creates the `hw-2` schema and `weather_documents` table automatically
# MAGIC 5. Stores weather documents for RAG/embedding use cases
# MAGIC 6. **AUTOMATIC:** Generates embeddings using SentenceTransformer during sync
# MAGIC 7. **AUTOMATIC:** Stores embeddings in `weather_embeddings` table with vector similarity search support
# MAGIC 8. Embeddings are generated in batches using `execute_values` for efficient writes

# COMMAND ----------

# DBTITLE 1,Install dependencies
# Install required packages
%pip uninstall -y psycopg2 psycopg2-binary
%pip install requests databricks-sdk sqlalchemy sentence-transformers torch --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Initialize Lakebase connection
import base64
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient
import psycopg2

w = WorkspaceClient()

def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="hw-db", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")

lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using OAuth token authentication as user: {db_user}\n")

# Create connection using psycopg2
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=10
    )
    cursor = conn.cursor()
    
    # Test the connection
    cursor.execute("SELECT version()")
    version = cursor.fetchone()[0]
    
    print(f"✓ Successfully connected to Lakebase using psycopg2")
    print(f"  PostgreSQL version: {version.split()[0]} {version.split()[1]}")
    print(f"  Schema: hw-2")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,Verify package imports
# Verify that the weather_app package is properly structured and importable
try:
    from weather_app import lakebase, sync_weather_data, WeatherFetcher, MAJOR_US_CITIES
    print("✓ Package structure verified!")
    print(f"  - weather_app.lakebase: {lakebase}")
    print(f"  - weather_app.sync_weather_data: {sync_weather_data}")
    print(f"  - weather_app.WeatherFetcher: {WeatherFetcher}")
    print(f"  - weather_app.MAJOR_US_CITIES: {len(MAJOR_US_CITIES)} cities defined")
    print("\n✅ All modules are properly packaged and importable!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure the weather_app/ folder exists with all required modules.")

# COMMAND ----------

# DBTITLE 1,Drop old table (if exists)
# Add missing columns to the existing table (if needed)
from weather_app import lakebase

try:
    lakebase.run_query('ALTER TABLE "hw-2".weather_documents ADD COLUMN IF NOT EXISTS location_lat DOUBLE PRECISION')
    lakebase.run_query('ALTER TABLE "hw-2".weather_documents ADD COLUMN IF NOT EXISTS location_lon DOUBLE PRECISION')
    lakebase.run_query('ALTER TABLE "hw-2".weather_documents ADD COLUMN IF NOT EXISTS event TEXT')
    lakebase.run_query('ALTER TABLE "hw-2".weather_documents ADD COLUMN IF NOT EXISTS effective_at TIMESTAMPTZ')
    lakebase.run_query('ALTER TABLE "hw-2".weather_documents ADD COLUMN IF NOT EXISTS payload JSONB')
    lakebase.run_query('ALTER TABLE "hw-2".weather_documents ADD COLUMN IF NOT EXISTS synced_at TIMESTAMPTZ')
    print("✓ Updated table schema with missing columns")
except Exception as e:
    print(f"⚠ Note: {e}")
    print("  (Table might not exist yet - will be created by sync)")

# COMMAND ----------

# DBTITLE 1,Configure embedding parameters
from weather_app import lakebase

print("Fixing weather_embeddings table ownership...")
print()

try:
    # Drop the existing table if it exists
    lakebase.run_query('DROP TABLE IF EXISTS "hw-2".weather_embeddings')
    print("✓ Dropped existing weather_embeddings table")
    print("  (The sync process will recreate it with correct ownership)")
except Exception as e:
    print(f"⚠ Note: {e}")
    print("  (Table might not exist or already accessible)")

print()
print("Now run Cell 7 to sync data and regenerate embeddings.")

# COMMAND ----------

# DBTITLE 1,Configure embedding parameters
# Restart Python to load the updated sync module
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Create embedding configuration widgets
# === SETUP: Run this cell once after Python restart ===
import os
import sys

# Configure Python path (done once for the entire notebook)
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    project_dir = os.path.dirname(notebook_path)
except:
    project_dir = os.getcwd()

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

print(f"✓ Python path configured: {project_dir}")
print()

# Create widgets for embedding configuration
dbutils.widgets.dropdown('embeddings_enabled', 'true', ['true', 'false'], 'Enable Embeddings')
dbutils.widgets.text('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2', 'Embedding Model')
dbutils.widgets.text('chunk_size', '800', 'Chunk Size')
dbutils.widgets.text('chunk_overlap', '100', 'Chunk Overlap')
dbutils.widgets.text('batch_size', '32', 'Batch Size')
dbutils.widgets.text('max_workers', '4', 'Max Workers')

print("✓ Embedding configuration widgets created")
print("  Use the dropdown and text boxes above to configure embedding settings")
print("  Note: Embedding dimensions are automatically determined by the model")

# COMMAND ----------

# DBTITLE 1,Sync weather data for all cities
from weather_app import sync_weather_data, get_embedding_dim, MAJOR_US_CITIES
from datetime import datetime
import time

# First, test the Lakebase connection
print("=" * 80)
print("TESTING LAKEBASE CONNECTION")
print("=" * 80)
print()

try:
    from weather_app import lakebase
    
    # Test connection with a simple query
    print("Testing connection to Lakebase...")
    test_result = lakebase.run_query("SELECT 1 as test")
    print(f"✓ Connection successful: {test_result}")
    print()
    
    # Fix table ownership if needed
    print("Checking table ownership...")
    role_result = lakebase.run_query("SELECT current_role")
    current_role = role_result[0]['current_role']
    
    table_owner_result = lakebase.run_query("""
        SELECT tableowner 
        FROM pg_tables 
        WHERE schemaname = 'hw-2' AND tablename = 'weather_documents'
    """)
    
    if table_owner_result and table_owner_result[0]['tableowner'] != current_role:
        print(f"Fixing table ownership (current role: {current_role})...")
        lakebase.run_query('ALTER TABLE "hw-2".weather_documents OWNER TO student')
        lakebase.run_query('ALTER TABLE "hw-2".weather_embeddings OWNER TO student')
        print("✓ Table ownership updated")
    else:
        print("✓ Table ownership is correct")
    print()
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print()
    print("TROUBLESHOOTING STEPS:")
    print("1. The Lakebase instance may have auto-suspended due to inactivity.")
    print("2. Go to Databricks UI → Lakebase section")
    print("3. Find your project: 'ai-data-engineer-bootcamp-v4'")
    print("4. Check the endpoint status and click to resume if paused")
    print("5. Wait 30-60 seconds for the endpoint to become active")
    print("6. Re-run this cell")
    print()
    print("Alternatively, the endpoint may need IP allowlist configuration.")
    print("Contact your workspace admin if the issue persists.")
    print()
    raise

# Read embedding configuration from widgets
embeddings_enabled = dbutils.widgets.get('embeddings_enabled') == 'true'
embedding_model = dbutils.widgets.get('embedding_model')
chunk_size = int(dbutils.widgets.get('chunk_size'))
chunk_overlap = int(dbutils.widgets.get('chunk_overlap'))
batch_size = int(dbutils.widgets.get('batch_size'))
max_workers = int(dbutils.widgets.get('max_workers'))

# Get embedding dimensions for the selected model
embedding_dim = get_embedding_dim(embedding_model)

print("=" * 80)
print("WEATHER DATA SYNC - 20 MAJOR US CITIES")
print("=" * 80)
print()
print(f"Embedding Configuration (from widgets):")
print(f"  Enabled: {embeddings_enabled}")
print(f"  Model: {embedding_model}")
print(f"  Dimensions: {embedding_dim}")
print(f"  Chunk size: {chunk_size} chars (overlap: {chunk_overlap})")
print(f"  Batch size: {batch_size}, Workers: {max_workers}")
print()

# Configure the sync - use cities from weather_app.MAJOR_US_CITIES
config = {
    "locations": list(MAJOR_US_CITIES.keys()),  # All 20 cities from weather_client
    "limit": 100,  # Max documents PER CITY (20 cities × 100 = up to 2,000 total)
    "include_alerts": True,
    "include_forecasts": True,
    "include_hourly": True,  # Include hourly forecasts
    "embeddings": {  # Automatic embedding generation (configured via widgets)
        "enabled": embeddings_enabled,
        "model_name": embedding_model,
        "embedding_dim": embedding_dim,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "batch_size": batch_size,
        "max_workers": max_workers,
    },
}

print(f"Starting sync for {len(config['locations'])} cities...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Run the sync
start_time = datetime.now()
result = sync_weather_data(config)
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print()
print("=" * 80)
print("📊 SYNC COMPLETE - SUMMARY")
print("=" * 80)
print()
print(f"⏱️  Duration: {duration:.2f} seconds")
print(f"📍 Cities synced: {len(config['locations'])}")
print()
print(f"📥 Documents fetched from API: {result['documents_fetched']}")
print(f"💾 Documents saved to database: {result['documents_upserted']}")
print(f"🤖 Embeddings generated: {result.get('chunks_embedded', 0)} chunks")
print()

if result['stats']:
    print("📈 Breakdown by Type:")
    print("-" * 80)
    for key, value in sorted(result['stats'].items()):
        print(f"  {key:20s}: {value:>4} documents")
    print("-" * 80)
    print()

if result['errors']:
    print(f"⚠️  Errors encountered: {len(result['errors'])}")
    print("\nFirst few errors:")
    for error in result['errors'][:5]:
        print(f"  - {error}")
else:
    print("✅ No errors - sync completed successfully!")

print()

# Show per-city breakdown
print("📍 Documents per City:")
print("-" * 80)
try:
    city_stats = lakebase.run_query("""
        SELECT 
            location,
            COUNT(*) as doc_count,
            COUNT(DISTINCT source_type) as doc_types
        FROM "hw-2".weather_documents
        GROUP BY location
        ORDER BY doc_count DESC
    """)
    
    for row in city_stats:
        print(f"  {row['location']:25s}: {row['doc_count']:>4} documents ({row['doc_types']} types)")
    print("-" * 80)
except Exception as e:
    print(f"  (Unable to fetch per-city stats: {e})")

print()
print("=" * 80)
print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Query stored data
# Query the database to see what we stored
from weather_app import lakebase

query_result = lakebase.run_query("""
    SELECT 
        source_type,
        COUNT(*) as count,
        COUNT(DISTINCT location) as unique_locations
    FROM "hw-2".weather_documents
    GROUP BY source_type
    ORDER BY count DESC
""")

print("Weather Documents by Type:")
print("=" * 60)
for row in query_result:
    print(f"{row['source_type']:20s}: {row['count']:4} documents from {row['unique_locations']:3} locations")

# Get total count
total_result = lakebase.run_query('SELECT COUNT(*) as count FROM "hw-2".weather_documents')
total = total_result[0]['count']

print("=" * 60)
print(f"Total documents: {total}")
print()

# Show sample documents
print("\nSample Weather Documents:")
print("=" * 80)

sample_result = lakebase.run_query("""
    SELECT 
        location,
        source_type,
        headline,
        issued_at
    FROM "hw-2".weather_documents
    ORDER BY issued_at DESC
    LIMIT 10
""")

for row in sample_result:
    print(f"Location: {row['location']}")
    print(f"Type: {row['source_type']} | {row['headline'][:70]}...")
    print(f"Issued: {row['issued_at']}")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Note: Automatic Embedding Generation
# MAGIC %md
# MAGIC ## 🤖 Automatic Embedding Generation
# MAGIC
# MAGIC **NEW:** Embeddings are now generated **automatically** during the sync process!
# MAGIC
# MAGIC ### What Changed:
# MAGIC - ✅ **Before:** Manual embedding generation in separate cells
# MAGIC - ✅ **Now:** Embeddings are automatically generated and stored during `sync_weather_data()` in **Cell 7**
# MAGIC
# MAGIC ### How It Works:
# MAGIC 1. When documents are synced, the system checks which ones need embeddings
# MAGIC 2. Text is chunked using a sliding window (default: 500 chars with 100 overlap)
# MAGIC 3. Embeddings are generated in batches using SentenceTransformer
# MAGIC 4. Efficient batch inserts using `psycopg2.extras.execute_values` with `%s::vector` cast
# MAGIC 5. Only new/unembedded documents are processed (incremental updates)
# MAGIC
# MAGIC ### Configuration:
# MAGIC You can customize embedding settings using **widgets** in **Cell 6**:
# MAGIC - `Enable Embeddings`: Turn automatic embeddings on/off (default: `true`)
# MAGIC - `Embedding Model`: Model to use (default: `"sentence-transformers/all-MiniLM-L6-v2"`)
# MAGIC - `Embedding Dimensions`: Vector dimensions (default: `384`)
# MAGIC - `Chunk Size`: Text chunk size in characters (default: `500`)
# MAGIC - `Chunk Overlap`: Overlap between chunks in characters (default: `100`)
# MAGIC - `Batch Size`: Batch size for generation (default: `32`)
# MAGIC - `Max Workers`: Parallel workers (default: `4`)
# MAGIC
# MAGIC These widget values are read by **Cell 7** and passed to `sync_weather_data()`.
# MAGIC
# MAGIC ### Performance:
# MAGIC - Uses `psycopg2` with `execute_values` for batch writes (NOT Spark JDBC)
# MAGIC - Casts embeddings as `%s::vector` for proper type handling
# MAGIC - ThreadPoolExecutor available for parallel processing if needed
# MAGIC - Only processes documents that don't have embeddings yet
# MAGIC
# MAGIC ### In app.py:
# MAGIC The Flask app also uses this automatic embedding generation. Configuration is in `app.yaml` under the `embeddings:` section.

# COMMAND ----------

# DBTITLE 1,Verify embedding statistics
from weather_app import lakebase

print("=" * 80)
print("📊 EMBEDDING STATISTICS")
print("=" * 80)
print()

try:
    # Check if embeddings table exists and has data
    stats = lakebase.run_query("""
        SELECT 
            COUNT(*) as total_chunks,
            COUNT(DISTINCT document_id) as embedded_docs,
            model_name,
            chunk_size,
            chunk_overlap,
            AVG(LENGTH(chunk_text)) as avg_chunk_length,
            MIN(created_at) as first_embedded,
            MAX(created_at) as last_embedded
        FROM "hw-2".weather_embeddings
        GROUP BY model_name, chunk_size, chunk_overlap
    """)
    
    if stats:
        for row in stats:
            print(f"🤖 Model: {row['model_name']}")
            print(f"📦 Total chunks: {row['total_chunks']:,}")
            print(f"📄 Documents embedded: {row['embedded_docs']:,}")
            print(f"📏 Chunk size: {row['chunk_size']} chars (overlap: {row['chunk_overlap']})")
            print(f"📐 Average chunk length: {row['avg_chunk_length']:.1f} chars")
            print(f"⏰ First embedded: {row['first_embedded']}")
            print(f"⏰ Last embedded: {row['last_embedded']}")
            
            # Calculate average chunks per document
            avg_chunks_per_doc = row['total_chunks'] / row['embedded_docs']
            print(f"📊 Average chunks per document: {avg_chunks_per_doc:.1f}")
            print()
        
        # Check coverage: how many documents have embeddings vs total documents
        coverage = lakebase.run_query("""
            SELECT 
                COUNT(DISTINCT d.id) as total_docs,
                COUNT(DISTINCT e.document_id) as embedded_docs,
                COUNT(DISTINCT d.id) - COUNT(DISTINCT e.document_id) as unembedded_docs
            FROM "hw-2".weather_documents d
            LEFT JOIN "hw-2".weather_embeddings e ON d.id = e.document_id
        """)
        
        if coverage:
            cov = coverage[0]
            pct = (cov['embedded_docs'] / cov['total_docs'] * 100) if cov['total_docs'] > 0 else 0
            print("📈 Coverage:")
            print(f"   Total documents: {cov['total_docs']:,}")
            print(f"   Embedded: {cov['embedded_docs']:,} ({pct:.1f}%)")
            print(f"   Not embedded: {cov['unembedded_docs']:,}")
            
            if cov['unembedded_docs'] > 0:
                print(f"\n⚠️  Note: {cov['unembedded_docs']} documents don't have embeddings yet.")
                print("   Run cell 7 again to embed them.")
            else:
                print("\n✅ All documents have embeddings!")
    else:
        print("⚠️  No embeddings found in the database yet.")
        print("   Run cell 7 to sync data and generate embeddings automatically.")
        
except Exception as e:
    if "does not exist" in str(e).lower():
        print("⚠️  Embeddings table doesn't exist yet.")
        print("   Run cell 7 to sync data and it will create the table automatically.")
    else:
        print(f"❌ Error querying embeddings: {e}")

print()
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Test vector similarity search
from weather_app import lakebase
from sentence_transformers import SentenceTransformer

# Use the same model that was used for embedding generation
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reuse cached model if available
if '_cached_model' in globals() and globals()['_cached_model'] is not None:
    model = globals()['_cached_model']
    print(f"✓ Reusing cached model: {EMBEDDING_MODEL}\n")
else:
    print(f"Loading model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder="/tmp/.cache/huggingface")
    globals()['_cached_model'] = model
    print(f"✓ Model loaded successfully\n")

# Test query
test_query = "severe weather alert thunderstorm"
print(f"Test query: '{test_query}'\n")

# Generate embedding for the query
query_embedding = model.encode(test_query).tolist()

# Find most similar chunks using cosine similarity
similarity_search_sql = """
    SELECT 
        d.location,
        d.source_type,
        d.headline,
        d.event,
        e.chunk_index,
        e.chunk_text,
        e.chunk_size,
        1 - (e.embedding <=> %s::vector) as similarity
    FROM "hw-2".weather_embeddings e
    JOIN "hw-2".weather_documents d ON e.document_id = d.id
    ORDER BY e.embedding <=> %s::vector
    LIMIT 10
"""

results = lakebase.run_query(similarity_search_sql, (query_embedding, query_embedding))

print("Top 10 most similar chunks:\n")
print('='*80)
for i, doc in enumerate(results, 1):
    print(f"\n{i}. Location: {doc['location']}")
    print(f"   Type: {doc['source_type']}")
    print(f"   Headline: {doc['headline']}")
    if doc['event']:
        print(f"   Event: {doc['event']}")
    print(f"   Chunk: {doc['chunk_index']} (size: {doc['chunk_size']} chars)")
    print(f"   Similarity: {doc['similarity']:.4f}")
    print(f"   Chunk text (first 150 chars): {doc['chunk_text'][:150]}...")
    print('-'*80)

# COMMAND ----------

# DBTITLE 1,Backfill embeddings for unembedded documents
from datetime import datetime
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from weather_app import lakebase, get_embedding_dim

print("=" * 80)
print("🔧 BACKFILLING EMBEDDINGS FOR UNEMBEDDED DOCUMENTS")
print("=" * 80)
print()

# Read embedding config from widgets
model_name = dbutils.widgets.get('embedding_model')
chunk_size = int(dbutils.widgets.get('chunk_size'))
chunk_overlap = int(dbutils.widgets.get('chunk_overlap'))
batch_size = int(dbutils.widgets.get('batch_size'))

# Get embedding dimensions for the selected model
embedding_dim = get_embedding_dim(model_name)

print(f"Embedding Configuration:")
print(f"  Model: {model_name}")
print(f"  Dimensions: {embedding_dim}")
print(f"  Chunk size: {chunk_size} (overlap: {chunk_overlap})")
print(f"  Batch size: {batch_size}")
print()

# Get all document IDs without embeddings for this config
print("Checking for documents without embeddings...")
unembedded_query = """
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

unembedded_ids = lakebase.run_query(unembedded_query, (model_name, chunk_size, chunk_overlap))
document_ids = [row['id'] for row in unembedded_ids]

if not document_ids:
    print("✅ All documents already have embeddings!")
else:
    print(f"Found {len(document_ids)} documents without embeddings\n")
    
    # Load the embedding model
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name, cache_folder="/tmp/.cache/huggingface")
    print(f"✓ Model loaded\n")
    
    # Fetch documents
    fetch_query = """
        SELECT 
            id,
            location,
            narrative_text,
            event
        FROM "hw-2".weather_documents
        WHERE id = ANY(%s)
    """
    
    documents = lakebase.run_query(fetch_query, (document_ids,))
    print(f"Fetched {len(documents)} documents for embedding\n")
    
    # Helper function to chunk text
    def create_chunks(text: str, size: int, overlap: int) -> list[str]:
        if not text or len(text) == 0:
            return []
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += (size - overlap)
            if end >= len(text):
                break
        return chunks
    
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
        chunks = create_chunks(full_text, chunk_size, chunk_overlap)
        
        # Store chunk info
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                'document_id': doc['id'],
                'chunk_index': chunk_idx,
                'chunk_text': chunk_text
            })
    
    if not all_chunks:
        print("⚠️  No text chunks to embed")
    else:
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"Average chunks per document: {len(all_chunks) / len(documents):.1f}\n")
        print(f"Starting backfill at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = datetime.now()
        
        # Generate embeddings in batches
        total_embedded = 0
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_texts = [chunk['chunk_text'] for chunk in batch]
            
            # Generate embeddings
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            
            # Prepare data for batch insert
            values_list = [
                (
                    chunk_info['document_id'],
                    chunk_info['chunk_index'],
                    chunk_info['chunk_text'],
                    embedding.tolist(),
                    model_name,
                    chunk_size,
                    chunk_overlap,
                    datetime.now()
                )
                for chunk_info, embedding in zip(batch, embeddings)
            ]
            
            # Use execute_values for batch insert
            insert_sql = """
                INSERT INTO "hw-2".weather_embeddings 
                    (document_id, chunk_index, chunk_text, embedding, model_name, chunk_size, chunk_overlap, created_at)
                VALUES %s
                ON CONFLICT (document_id, chunk_index, model_name, chunk_size, chunk_overlap) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
            """
            
            with lakebase.get_connection() as conn:
                with conn.cursor() as cur:
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
            print(f"Batch {batch_num}/{total_batches}: {len(batch)} chunks ({total_embedded}/{len(all_chunks)} total)")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print()
        print("=" * 80)
        print("✅ BACKFILL COMPLETE")
        print("=" * 80)
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📄 Documents processed: {len(documents)}")
        print(f"🤖 Chunks embedded: {total_embedded}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

print()
print("Now run Cell 11 to verify all documents have embeddings.")