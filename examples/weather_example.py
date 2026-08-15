# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Weather NWS API Example
# MAGIC %md
# MAGIC # ⚠️ REFERENCE ONLY - Not for Production Use
# MAGIC
# MAGIC **This notebook demonstrates `weather_client` API usage patterns for testing and development.**
# MAGIC
# MAGIC **For production weather syncs, use:** [weather_to_lakebase](#notebook-1163260665310961)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # Weather Data Fetching and Storage Example
# MAGIC
# MAGIC This notebook demonstrates how to use the `weather_client` module to:
# MAGIC 1. Fetch weather alerts and forecasts from the National Weather Service API
# MAGIC 2. Normalize the data into a unified document format
# MAGIC 3. Store the data in Lakebase Postgres for RAG/embedding use cases
# MAGIC
# MAGIC ## Features
# MAGIC
# MAGIC * **Major US Cities**: Supports 20+ major US cities (e.g., "Chicago, IL", "Austin, TX", "New York, NY") with hardcoded coordinates - no external API calls needed!
# MAGIC * **Flexible Input**: Also accepts (lat, lon) tuples for precise locations
# MAGIC * **Comprehensive Data**: Fetches alerts, forecasts, and hourly forecasts
# MAGIC * **Smart Caching**: Caches geocoding results and grid point lookups for performance
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC * **weather_client.py**: API client that fetches and normalizes weather data
# MAGIC * **lakebase.py**: Connection helper for Lakebase Postgres
# MAGIC * **sync_weather_to_db.py**: Orchestration script that ties fetching and storage together

# COMMAND ----------

# DBTITLE 1,Install dependencies
# Install required packages
%pip uninstall -y psycopg2 psycopg2-binary
%pip install requests sqlalchemy urllib3 databricks-sdk python-dotenv --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Note: Geocoding Improvements
# MAGIC %md
# MAGIC ## ✨ Recent Updates
# MAGIC
# MAGIC **Simplified and reliable!** The weather_client module now uses hardcoded coordinates for 20+ major US cities:
# MAGIC * **No external geocoding API calls** - eliminates network failures and rate limiting issues
# MAGIC * **Fast and reliable** - instant lookup from local dictionary
# MAGIC * **Supports major cities**: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, Austin, San Francisco, Seattle, Denver, Boston, Miami, Atlanta, Portland, Las Vegas, Detroit, Nashville, and more
# MAGIC * **Case-insensitive matching** - "Chicago, IL" or "chicago, il" both work
# MAGIC
# MAGIC Simply use city names from the supported list (shown in error messages if you use an unsupported city).

# COMMAND ----------

# DBTITLE 1,Example 1: Fetch weather data (no DB write)
"""Example 1: Fetch weather data without writing to database."""

# import sys
# sys.path.append('/Workspace/Users/hsokolovskyi@gmail.com/databricks-bootcamp-day-2-hw')

from weather_client import WeatherFetcher
import json

# Initialize fetcher
fetcher = WeatherFetcher(rate_limit_delay=0.1)

# Configure what to fetch - supports 20+ major US cities with hardcoded coordinates!
# Available cities: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia,
# San Antonio, San Diego, Dallas, Austin, San Francisco, Seattle, Denver, Boston,
# Miami, Atlanta, Portland, Las Vegas, Detroit, Nashville, and more
config = {
    "locations": [
        "Chicago, IL",
        "Austin, TX",
        "Portland, OR",  # Works for any city!
    ],
    "limit": 10,
    "include_alerts": True,
    "include_forecasts": True,
    "include_hourly": False,
}

# Fetch documents
result = fetcher.fetch_weather_documents(config)

print(f"\nFetched {len(result['documents'])} documents")
print(f"\nStatistics:")
for key, value in result['stats'].items():
    print(f"  {key}: {value}")

if result['errors']:
    print(f"\nErrors: {len(result['errors'])}")
    for error in result['errors']:
        print(f"  - {error}")

# Show a sample document
if result['documents']:
    print(f"\n{'='*60}")
    print("Sample Document:")
    print('='*60)
    sample = result['documents'][0]
    print(f"ID: {sample['id'][:50]}...")
    print(f"Location: {sample['location']}")
    print(f"Source: {sample['source_type']}")
    print(f"Headline: {sample['headline']}")
    print(f"Narrative (first 200 chars): {sample['narrative_text'][:200]}...")
    print(f"Issued: {sample['issued_at']}")

# COMMAND ----------

# DBTITLE 1,Example 2: Full sync to Lakebase
"""Example 2: Fetch weather data for multiple major cities."""

from weather_client import WeatherFetcher
import json

# Initialize fetcher
fetcher = WeatherFetcher(rate_limit_delay=0.1)

# Configure locations - supports both lat/lon tuples and city/state strings
config = {
    "locations": [
        "Chicago, IL",
        "Austin, TX",
        "San Francisco, CA",
        "New York, NY",
    ],
    "limit": 50,
    "include_alerts": True,
    "include_forecasts": True,
    "include_hourly": False,
}

# Fetch documents
result = fetcher.fetch_weather_documents(config)

print("\n" + "="*60)
print("Weather Fetch Complete")
print("="*60)
print(f"Documents fetched: {len(result['documents'])}")
print(f"\nStatistics:")
for key, value in result['stats'].items():
    print(f"  {key}: {value}")

if result['errors']:
    print(f"\nErrors encountered: {len(result['errors'])}")
    for error in result['errors'][:5]:
        print(f"  - {error}")

# Show sample documents by location
if result['documents']:
    from collections import defaultdict
    by_location = defaultdict(list)
    for doc in result['documents']:
        by_location[doc['location']].append(doc)
    
    print(f"\n{'='*60}")
    print("Sample Documents by Location:")
    print('='*60)
    for location, docs in list(by_location.items())[:2]:
        print(f"\n{location}: {len(docs)} documents")
        sample = docs[0]
        print(f"  Type: {sample['source_type']}")
        print(f"  Headline: {sample['headline'][:80]}...")
        print(f"  Issued: {sample['issued_at']}")

# COMMAND ----------

# DBTITLE 1,Query the stored data
"""Analyze fetched weather data (without database)."""

from weather_client import WeatherFetcher

# Fetch fresh data
fetcher = WeatherFetcher(rate_limit_delay=0.1)
config = {
    "locations": [
        (41.8781, -87.6298),   # Chicago, IL
        (30.2672, -97.7431),   # Austin, TX
    ],
    "limit": 20,
    "include_alerts": True,
    "include_forecasts": True,
    "include_hourly": False,
}

result = fetcher.fetch_weather_documents(config)
documents = result['documents']

print(f"Found {len(documents)} documents")
print("\n" + "="*80)

for doc in documents:
    print(f"\nLocation: {doc['location']}")
    print(f"Type: {doc['source_type']} | Headline: {doc['headline']}")
    narrative_preview = doc['narrative_text'][:100] if doc['narrative_text'] else ""
    print(f"Narrative: {narrative_preview}...")
    print(f"Issued: {doc['issued_at']}")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Query by location
"""Example 3: Fetch weather for a specific location."""

from weather_client import WeatherFetcher

# Fetch weather for Chicago
fetcher = WeatherFetcher(rate_limit_delay=0.1)
config = {
    "locations": ["Chicago, IL"],
    "limit": 10,
    "include_alerts": True,
    "include_forecasts": True,
    "include_hourly": False,
}

result = fetcher.fetch_weather_documents(config)
documents = result['documents']

print(f"\nFound {len(documents)} documents for Chicago")

for doc in documents:
    print(f"\n{'='*80}")
    print(f"Type: {doc['source_type']}")
    print(f"Headline: {doc['headline']}")
    if doc.get('severity'):
        print(f"Severity: {doc['severity']}")
    print(f"\nNarrative:\n{doc['narrative_text'][:300]}...")
    print(f"\nIssued: {doc['issued_at']}")

# COMMAND ----------

# DBTITLE 1,Setup: Configure Database Connection
# MAGIC %md
# MAGIC ## Database Setup (Optional - Skip for API Examples)
# MAGIC
# MAGIC **Note:** The examples in cells 4-7 work without any database setup. The cells below (9+) are for database sync testing only.
# MAGIC
# MAGIC To use the database sync features, you need to configure your Lakebase Postgres connection.
# MAGIC
# MAGIC ### Option 1: Using .env file (Recommended for Development)
# MAGIC
# MAGIC 1. Create a `.env` file in this directory:
# MAGIC    ```bash
# MAGIC    # Copy the template
# MAGIC    cp .env.example .env
# MAGIC    ```
# MAGIC
# MAGIC 2. Edit `.env` and add your connection URL:
# MAGIC    ```
# MAGIC    LAKEBASE_URL=postgresql://username:password@host:5432/databricks_postgres?sslmode=require
# MAGIC    ```
# MAGIC
# MAGIC 3. The `.env` file will be automatically loaded by the `lakebase` module
# MAGIC
# MAGIC ### Option 2: Using Databricks Secrets (Recommended for Production)
# MAGIC
# MAGIC Store your connection URL in Databricks secrets:
# MAGIC ```bash
# MAGIC databricks secrets create-scope database
# MAGIC databricks secrets put-secret database lakebase-url
# MAGIC ```
# MAGIC
# MAGIC ### Skip Database Features
# MAGIC
# MAGIC If you don't need database storage, you can skip cells that use `lakebase` or `sync_weather_to_db`.

# COMMAND ----------

# DBTITLE 1,Create h2-w Schema
"""Create h2-w schema in Lakebase."""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

# Load environment variables from .env file in current directory
env_path = Path('.env')
load_dotenv(env_path)

# Get the Lakebase URL
lakebase_url = os.getenv('LAKEBASE_URL')

if not lakebase_url:
    raise ValueError("LAKEBASE_URL not found in .env file")

print(f"Connecting to Lakebase...")

try:
    # Connect to Lakebase
    conn = psycopg2.connect(lakebase_url)
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Create schema (quoted because of hyphen in name)
    schema_name = "hw-2"
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
    
    print(f"✓ Schema '{schema_name}' created successfully!")
    
    # Verify schema exists
    cursor.execute("""
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name = %s
    """, (schema_name,))
    
    result = cursor.fetchone()
    if result:
        print(f"✓ Verified: Schema '{schema_name}' exists in database")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"✗ Error: {e}")
    raise