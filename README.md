# Weather NWS Data Pipeline

A complete weather data solution that fetches alerts and forecasts from the National Weather Service API, stores them in Lakebase Postgres, generates vector embeddings for semantic search, and provides a web dashboard for visualization and monitoring.

## Data Source Selection

**Data Source:** National Weather Service (NWS) API - https://api.weather.gov/

**Why NWS API?**
* **No API Key Required**: Free public API with no authentication needed
* **Real-time Official Data**: Authoritative weather information from NOAA
* **Comprehensive Coverage**: Alerts, daily forecasts, and hourly forecasts for all US locations
* **Structured JSON**: Well-documented API with consistent response format
* **RAG-Friendly Content**: Narrative text fields perfect for semantic search (e.g., alert descriptions, forecast narratives)
* **Location Flexibility**: Supports both lat/lon coordinates and grid points

**Alternative Considered:** OpenWeather API - requires API key and has usage limits

## Schema Design Decisions

### Weather Documents Table Schema

**Key Columns:**
* `id` (TEXT PRIMARY KEY): Composite key from `{location}:{source_type}:{issued_at}` for stable deduplication
* `location`, `location_lat`, `location_lon`: Human-readable city name + coordinates for filtering and mapping
* `source_type`: Categorizes as 'alert', 'forecast', or 'hourly_forecast' for easy filtering
* `headline`: Short summary text for display
* `narrative_text`: Full description - **primary RAG target** (averaged 500-1000 chars per document)
* Temporal fields (`issued_at`, `effective_at`, `expires_at`): Critical for time-series queries and freshness filtering
* `payload` (JSONB): Complete API response preserved for future feature extraction
* `synced_at`: Tracks data pipeline execution timestamps

**Design Rationale:**
* Normalized structure balances queryability with flexible JSONB storage
* Separate lat/lon columns enable spatial queries
* Text fields optimized for semantic search use case

### Embedding Configuration

**Model Selection:** `sentence-transformers/all-MiniLM-L6-v2`
* **Dimensions:** 384 (good balance of quality and performance)
* **Rationale:** Fast inference, low memory footprint, proven quality for retrieval tasks
* **Alternatives Tested:** all-mpnet-base-v2 (768 dims) - higher quality but slower

**Chunking Strategy:**
* **Method:** Sliding window with overlap
* **Chunk Size:** 800 characters (adjusted from initial 500)
  * Rationale: Weather documents averaged 100-200 chars, so most fit in a single chunk
  * Larger chunks preserve more context for longer forecast narratives
* **Chunk Overlap:** 100 characters (12.5% of chunk size)
  * Rationale: Captures context across boundaries while minimizing redundancy
  * Critical for multi-sentence alerts that might be split
* **Result:** Average 1.0 chunks per document (most weather data is concise)

**Vector Index:**
* **Type:** HNSW (Hierarchical Navigable Small World)
* **Metric:** Cosine similarity (`vector_cosine_ops`)
* **Rationale:** Approximate nearest neighbor search with 10-100x speedup over brute force

### Embeddings Table Schema

**Key Design Choices:**
* `embedding_id` (SERIAL): Auto-incrementing primary key for insert performance
* `document_id` + `chunk_index`: Composite unique constraint for chunk deduplication
* Config tracking (`model_name`, `chunk_size`, `chunk_overlap`): Enables multi-config experimentation
* `created_at`: Timestamp for debugging and incremental updates
* **Foreign Key with CASCADE DELETE**: Ensures embeddings are automatically cleaned up when source documents are deleted

## How to Run the Sync → Embed → Search Pipeline End-to-End

### Option 1: Using the Notebook (Recommended for Learning)

1. **Open the notebook**: [weather_to_lakebase](#notebook-1163260665310961)

2. **Install dependencies** (Cell 2):
   ```python
   %pip install requests databricks-sdk sqlalchemy sentence-transformers torch --quiet
   dbutils.library.restartPython()
   ```

3. **Configure database connection** (Cell 3):
   * Uses Databricks secrets: `hw-db/lakebase-url`
   * Or set `LAKEBASE_URL` in `.env` file for local development

4. **Configure embedding parameters** (Cell 8):
   * Use the widgets to set:
     * Enable Embeddings: `true`
     * Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
     * Chunk Size: `800`
     * Chunk Overlap: `100`
     * Batch Size: `32`

5. **Run the sync** (Cell 9):
   ```python
   from weather_app import sync_weather_data
   
   result = sync_weather_data(
       config={
           "locations": ["Chicago, IL", "New York, NY"],  # 20 major cities by default
           "limit": 50,
           "include_alerts": True,
           "include_forecasts": True,
           "include_hourly": True,
       },
       embedding_config={...}  # Auto-configured from widgets
   )
   ```
   * This single cell:
     * Fetches weather data from NWS API
     * Creates schema and tables if needed
     * Upserts documents to Lakebase
     * Generates embeddings automatically
     * Creates HNSW vector index

6. **Verify the data** (Cell 10):
   * Check document counts by type
   * View sample documents

7. **Check embedding coverage** (Cell 12):
   * See statistics on embeddings generated
   * Verify 100% coverage

8. **Run a semantic search** (Cell 13):
   ```python
   from sentence_transformers import SentenceTransformer
   from weather_app import lakebase
   
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   query_embedding = model.encode("severe thunderstorm alert").tolist()
   
   results = lakebase.run_query("""
       SELECT d.location, d.headline, e.chunk_text,
              1 - (e.embedding <=> %s::vector) as similarity
       FROM "hw-2".weather_embeddings e
       JOIN "hw-2".weather_documents d ON e.document_id = d.id
       ORDER BY e.embedding <=> %s::vector
       LIMIT 10
   """, (query_embedding, query_embedding))
   ```

### Option 2: Using the Flask Web Dashboard (Recommended for Production)

1. **Configure** `app.yaml`:
   ```yaml
   weather:
     locations:
       - "New York, NY"
       - "Chicago, IL"
     limit: 50
     include_alerts: true
   
   embeddings:
     enabled: true
     model_name: "sentence-transformers/all-MiniLM-L6-v2"
     chunk_size: 800
     chunk_overlap: 100
   ```

2. **Run the dashboard**:
   ```bash
   python app.py
   ```

3. **Trigger sync via UI**:
   * Visit http://localhost:8000
   * Click the "Sync Data" button
   * Embeddings are generated automatically during sync

4. **Or trigger via API**:
   ```bash
   curl -X POST http://localhost:8000/sync
   ```

### What Happens During Each Sync:

1. **Fetch**: Calls NWS API for each location (alerts + forecasts + hourly)
2. **Normalize**: Transforms API responses into consistent document format
3. **Upsert**: Inserts/updates documents in `weather_documents` table
4. **Chunk**: Splits narrative text using sliding window (800 chars, 100 overlap)
5. **Embed**: Generates 384-dim vectors using SentenceTransformer (batch size 32)
6. **Store**: Inserts embeddings into `weather_embeddings` table with proper type casting
7. **Index**: HNSW index automatically updates for fast similarity search

### Performance Notes:

* **First run**: ~2-3 minutes for 20 cities (~2000 documents + embeddings)
* **Incremental runs**: <30 seconds (only new/updated documents embedded)
* **Search queries**: <100ms with HNSW index

## Known Limitations & Future Improvements

### Current Limitations

1. **US-Only Coverage**
   * NWS API only covers United States locations
   * Could integrate OpenWeather or other APIs for international coverage

2. **No Multi-Tenancy**
   * Single shared schema (`hw-2`) for all data
   * Would need user-specific schemas for production multi-tenant deployment

3. **Limited Error Recovery**
   * API failures for individual cities don't retry automatically
   * Would benefit from exponential backoff and dead letter queue

4. **No Real-time Updates**
   * Poll-based sync (manual or scheduled)
   * Could use WebSocket or webhook subscriptions for push-based updates

5. **Basic Chunking**
   * Fixed-size sliding window doesn't respect sentence boundaries
   * Could use semantic chunking (sentence/paragraph-aware) for better context preservation

6. **Single Embedding Model**
   * Locked to one model per run (changing model requires re-embedding all docs)
   * Could support multiple embedding models side-by-side for A/B testing

7. **No Query Reranking**
   * Vector search returns raw HNSW results
   * Could add cross-encoder reranking for better precision

8. **Limited Observability**
   * Basic logging only
   * Would benefit from metrics (sync duration, API latency, embedding throughput) and alerts

9. **Vector SQL Construction in Search Endpoint**
   * The `/weather/search` endpoint (app.py lines 1543-1563) currently inlines the query vector into SQL as a string literal
   * Should use parameterized queries with `%s::vector` for consistency with embedding insert operations
   * Current approach works but generates large SQL strings and is less maintainable

### Given More Time, I Would:

1. **Add Hybrid Search**
   * Combine vector similarity with keyword search (BM25)
   * Implement RRF (Reciprocal Rank Fusion) for ranking

2. **Implement RAG Pipeline**
   * Add LLM integration for natural language Q&A
   * Context retrieval from top-K similar chunks
   * Response generation with citation to source documents

3. **Improve Data Quality**
   * Deduplication logic for near-duplicate forecasts
   * Data validation and schema enforcement
   * Historical data retention policies (archive old forecasts)

4. **Add Analytics Layer**
   * Trend analysis (alert frequency, severe weather patterns)
   * Comparative analysis (city-to-city, month-over-month)
   * Materialized views for common aggregations

5. **Enhance Web Dashboard**
   * Interactive map visualization with alert overlays
   * Real-time WebSocket updates
   * Search interface for semantic queries
   * Export functionality (CSV, JSON)

6. **Production Hardening**
   * Comprehensive error handling and retry logic
   * Rate limiting and circuit breakers for NWS API
   * Connection pooling for Lakebase
   * Async embedding generation with job queue (Celery)

7. **Testing & CI/CD**
   * Unit tests for API client, chunking, embedding logic
   * Integration tests with mock Lakebase
   * End-to-end tests for sync pipeline
   * Automated deployment via Declarative Automation Bundles

8. **Add More Data Sources**
   * NOAA historical data archives
   * Satellite imagery and radar data
   * Social media weather reports for real-time validation

## Features

* **Flexible Location Input**: Accepts both lat/lon tuples `(41.88, -87.63)` and city/state strings `"Chicago, IL"`
* **Major US Cities Support**: Pre-configured coordinates for 20+ major cities (no external geocoding needed)
* **Comprehensive Data**: Fetches weather alerts, daily forecasts, and hourly forecasts
* **Smart Caching**: Caches geocoding results and grid point lookups for performance
* **Database Storage**: Persistence to Lakebase Postgres with automatic upsert logic
* **Vector Embeddings**: Generate embeddings with sliding window chunking for semantic search
* **Configurable Chunking**: Widget-based controls for chunk size, overlap, and embedding model
* **Batch Processing**: Efficient batch embedding generation (32 chunks at a time)
* **Model Caching**: Reuse embedding models across notebook runs for faster execution
* **Vector Search**: HNSW index for fast similarity search over weather documents
* **Web Dashboard**: Flask application with real-time sync button and responsive UI
* **YAML Configuration**: Easy customization of cities, sync options, and display settings
* **Credential Flexibility**: Supports both `.env` files (development) and Databricks secrets (production)
* **Modular Package Structure**: Clean Python package for easy imports and reusability

## Quick Start

### 1. Install Dependencies

```bash
%pip install -r requirements.txt
```

Or in a notebook:
```python
%pip install requests psycopg2-binary sqlalchemy urllib3 databricks-sdk python-dotenv flask pyyaml sentence-transformers --quiet
```

### 2. Configure Database (Optional)

If you want to persist data to a database, choose one of these methods:

#### Option A: Using .env file (Recommended for Development)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Lakebase Postgres connection URL:
   ```
   LAKEBASE_URL=postgresql://username:password@host:5432/databricks_postgres?sslmode=require
   ```

3. The module will automatically load credentials from `.env`

#### Option B: Using Databricks Secrets (Recommended for Production)

```bash
databricks secrets create-scope database
databricks secrets put-secret database lakebase-url
```

### 3. Run Examples

See the `weather_example` notebook for complete examples:

1. **Fetch without database**: Simple API fetch and display
2. **Sync to database**: Fetch and persist to Lakebase Postgres
3. **Query by location**: Query stored documents for specific cities
4. **Statistics**: Analyze stored weather data

## Project Structure

```
├── weather_app/            # Main package directory
│   ├── __init__.py         # Package initialization and exports
│   ├── weather_client.py   # Core NWS API client and normalization
│   ├── sync_weather_to_db.py  # Database sync orchestration with embeddings
│   └── lakebase.py         # Lakebase Postgres connection helper
├── app.py                  # Flask web dashboard application
├── app.yaml                # Dashboard configuration (cities, sync, display)
├── requirements.txt        # Python dependencies
├── .env                    # Local credentials (not tracked in git)
├── .gitignore              # Protects .env from version control
└── weather_to_lakebase     # Complete workflow notebook (Databricks notebook)
```

## Module Usage

### Basic Fetching (No Database)

```python
from weather_nws import WeatherFetcher

fetcher = WeatherFetcher(rate_limit_delay=0.1)

config = {
    "locations": [
        "Chicago, IL",           # City/state string
        (30.2672, -97.7431),    # Or lat/lon tuple
    ],
    "limit": 50,
    "include_alerts": True,
    "include_forecasts": True,
    "include_hourly": False,
}

result = fetcher.fetch_weather_documents(config)
print(f"Fetched {len(result['documents'])} documents")
```

### Sync to Database

```python
from sync_weather_to_db import sync_weather_data

config = {
    "locations": [(41.88, -87.63), (30.27, -97.74)],
    "limit": 50,
    "include_alerts": True,
    "include_forecasts": True,
}

result = sync_weather_data(config)
print(f"Upserted {result['documents_upserted']} documents")
```

### Query Stored Data

```python
import lakebase

query = """
    SELECT location, headline, narrative_text
    FROM weather_documents
    WHERE location LIKE %s
    ORDER BY issued_at DESC
    LIMIT 10
"""

rows = lakebase.run_query(query, ('%Chicago%',))
for row in rows:
    print(f"{row['location']}: {row['headline']}")
```

### Vector Embeddings and Semantic Search

The pipeline includes vector embedding generation with sliding window chunking for semantic search:

```python
from weather_app import lakebase
from sentence_transformers import SentenceTransformer

# Configure embedding parameters (or use widgets in notebook)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlapping characters

# Load model with caching
model = SentenceTransformer(EMBEDDING_MODEL, cache_folder="/tmp/.cache/huggingface")

# Generate embeddings for unprocessed documents
# (See Cell 11 in weather_to_lakebase notebook for complete implementation)

# Search for similar content
query = "severe weather thunderstorm alert"
query_embedding = model.encode(query).tolist()

results = lakebase.run_query("""
    SELECT 
        d.location,
        d.headline,
        e.chunk_text,
        1 - (e.embedding <=> %s::vector) as similarity
    FROM "hw-2".weather_embeddings e
    JOIN "hw-2".weather_documents d ON e.document_id = d.id
    ORDER BY e.embedding <=> %s::vector
    LIMIT 10
""", (query_embedding, query_embedding))

for doc in results:
    print(f"{doc['location']}: {doc['headline']} (similarity: {doc['similarity']:.4f})")
```

#### Embedding Configuration Options

**Supported Models:**
* `sentence-transformers/all-MiniLM-L6-v2` (384 dims) - Fast, good quality
* `sentence-transformers/all-mpnet-base-v2` (768 dims) - Better quality
* `BAAI/bge-small-en-v1.5` (384 dims) - Optimized for retrieval
* `BAAI/bge-base-en-v1.5` (768 dims) - Balanced performance
* `BAAI/bge-large-en-v1.5` (1024 dims) - Highest quality

**Chunking Strategy:**
* **Sliding window**: Splits long documents into overlapping chunks
* **Chunk size**: Typical range 300-1000 characters (500 recommended)
* **Overlap**: Typically 10-20% of chunk size (100 chars for 500-char chunks)
* **Benefits**: Captures context across chunk boundaries, improves recall

**Performance Features:**
* Batch processing: 32 chunks embedded at once for efficiency
* Model caching: Reuses loaded model across notebook cells
* Smart skip: Only embeds documents not yet processed with current config
* HNSW index: Fast approximate nearest neighbor search

## Flask Web Dashboard

The project includes a Flask web application for visualizing and syncing weather data.

### Features

* **Latest Documents View**: Displays the 3 most recent weather documents for each tracked city
* **One-Click Sync**: Button to trigger fresh data fetch from NWS API
* **Real-time Statistics**: Total documents, cities tracked, and last update time
* **Responsive Design**: Modern gradient UI that works on all screen sizes
* **Auto-refresh**: Page automatically reloads every 5 minutes
* **REST API**: Endpoints for sync, health check, and statistics

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install flask pyyaml
   ```

2. **Configure cities** (edit `app.yaml`):
   ```yaml
   weather:
     locations:
       - "New York, NY"
       - "Los Angeles, CA"
       - "Chicago, IL"
   ```

3. **Run the dashboard**:
   ```bash
   python app.py
   ```

4. **Access**: Open http://localhost:8000 in your browser (default port from app.yaml)

### API Endpoints

* `GET /` - Main dashboard page
* `POST /weather/sync` - Trigger weather data sync with automatic embedding generation
* `POST /weather/search` - Semantic search over weather embeddings using vector similarity
* `GET /city/<city_name>` - Detailed view for a specific city
* `GET /health` - Health check
* `GET /api/stats` - Detailed statistics (JSON)

### Configuration

Edit `app.yaml` to customize:

```yaml
app:
  host: "0.0.0.0"
  port: 5000
  debug: false

database:
  schema: "hw-2"
  table: "weather_documents"

weather:
  locations:
    - "New York, NY"
    - "Chicago, IL"
  limit: 500
  include_alerts: true
  include_forecasts: true
  include_hourly: true

dashboard:
  title: "Weather Data Dashboard"
  documents_per_city: 3
  auto_refresh_minutes: 5
```

### Programmatic API Access

**Trigger sync from code:**

```python
import requests

# Sync specific cities with custom limit
response = requests.post('http://localhost:8000/weather/sync', json={
    "locations": ["Chicago, IL", "Austin, TX"],
    "limit": 50
})
result = response.json()

if result['success']:
    print(f"Synced! Fetched: {result['documents_fetched']}, Upserted: {result['documents_upserted']}")
    if result.get('chunks_embedded'):
        print(f"Embeddings: {result['chunks_embedded']} chunks")
```

**Semantic search from code:**

```python
import requests

# Search for weather events
response = requests.post('http://localhost:8000/weather/search', json={
    "query": "severe thunderstorm warning",
    "top_k": 5
})
result = response.json()

if result['success']:
    for item in result['results']:
        print(f"{item['location']}: {item['headline']} (similarity: {item['similarity']:.2%})")
```

## Credential Priority

The `lakebase` module looks for credentials in this order:

1. **LAKEBASE_URL environment variable** (from `.env` or shell)
2. **Databricks secrets** (scope: `database`, key: `lakebase-url`)

This allows seamless transition between local development and production deployment.

## Security Notes

* The `.env` file is automatically excluded from git via `.gitignore`
* Never commit actual credentials to version control
* Use `.env` for local development only
* Always use Databricks secrets for production workspaces

## Database Schema

### Weather Documents Table

The `weather_documents` table stores normalized weather data:

```sql
CREATE TABLE weather_documents (
    id TEXT PRIMARY KEY,                    -- Stable deduplication key
    location TEXT NOT NULL,                 -- City/area name
    location_lat DOUBLE PRECISION,          -- Latitude
    location_lon DOUBLE PRECISION,          -- Longitude
    source_type TEXT NOT NULL,              -- 'alert' | 'forecast' | 'hourly_forecast'
    headline TEXT,                          -- Short summary
    event TEXT,                             -- Event type (for alerts)
    severity TEXT,                          -- Severity level (for alerts)
    narrative_text TEXT,                    -- Full description (RAG-ready)
    issued_at TIMESTAMPTZ,                  -- When issued
    effective_at TIMESTAMPTZ,               -- When effective (alerts)
    expires_at TIMESTAMPTZ,                 -- When expires (alerts)
    payload JSONB,                          -- Complete API response
    synced_at TIMESTAMPTZ NOT NULL          -- When synced to DB
);
```

### Weather Embeddings Table

The `weather_embeddings` table stores vector embeddings with sliding window chunks:

```sql
CREATE TABLE weather_embeddings (
    embedding_id SERIAL PRIMARY KEY,        -- Auto-incrementing ID
    document_id TEXT NOT NULL,              -- Foreign key to weather_documents
    chunk_index INTEGER NOT NULL,           -- Position of chunk in document (0-based)
    chunk_text TEXT NOT NULL,               -- The actual text chunk
    embedding vector(384) NOT NULL,         -- Vector embedding (dimension varies by model)
    model_name TEXT NOT NULL,               -- Embedding model used
    chunk_size INTEGER NOT NULL,            -- Chunk size in characters
    chunk_overlap INTEGER NOT NULL,         -- Overlap size in characters
    created_at TIMESTAMPTZ DEFAULT NOW(),   -- When embedding was created
    UNIQUE(document_id, chunk_index),       -- One chunk per position per document
    FOREIGN KEY (document_id) 
        REFERENCES weather_documents(id) 
        ON DELETE CASCADE
);

-- Vector similarity search index (HNSW for fast approximate search)
CREATE INDEX idx_weather_embeddings_vector 
    ON weather_embeddings 
    USING hnsw (embedding vector_cosine_ops);

-- Index for joining with documents
CREATE INDEX idx_weather_embeddings_document_id 
    ON weather_embeddings(document_id);

-- Index for model and config lookups
CREATE INDEX idx_weather_embeddings_config 
    ON weather_embeddings(model_name, chunk_size, chunk_overlap);
```

**Key Features:**
* **Multi-chunk support**: Long documents split into multiple overlapping chunks
* **Config tracking**: Stores chunk_size and chunk_overlap used for generation
* **Model versioning**: Tracks which embedding model was used
* **Fast search**: HNSW index for efficient cosine similarity queries
* **Cascading deletes**: Embeddings auto-deleted when source document is removed

## Troubleshooting

### ModuleNotFoundError: No module named 'dotenv'

Install python-dotenv:
```bash
%pip install python-dotenv
```

### ValueError: Could not find Lakebase credentials

Make sure you either:
1. Created a `.env` file with `LAKEBASE_URL`
2. Set the `LAKEBASE_URL` environment variable
3. Configured Databricks secrets

### Census Geocoding API returns 0 matches

The US Census Geocoding API requires street addresses, not just city/state. Use lat/lon tuples instead:
```python
config = {
    "locations": [
        (41.8781, -87.6298),   # Chicago coordinates
    ],
    ...
}
```

## License

MIT
