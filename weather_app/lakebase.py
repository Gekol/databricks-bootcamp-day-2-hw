"""
Lakebase (Databricks-managed Postgres) connection helper.

Supports two methods for providing credentials:
1. .env file (recommended for local development): Create a .env file with LAKEBASE_URL
2. Databricks secrets (recommended for production): Store in secret scope

Connection URL format:
postgresql://role:password@host:5432/databricks_postgres?sslmode=require
"""

import base64
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in the current directory or parent directories
    dotenv_path = Path(__file__).parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        print(f"✓ Loaded credentials from {dotenv_path}")
    else:
        # Try loading from current working directory
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed


def _lakebase_url() -> str:
    """
    Fetch Lakebase connection URL from environment or Databricks secrets.
    
    Priority order:
    1. LAKEBASE_URL environment variable (from .env or shell)
    2. Databricks secret (decoded from base64)
    
    Returns:
        Connection URL string
        
    Raises:
        ValueError: If no credentials found in either location
    """
    # Try environment variable first (from .env or shell)
    url = os.environ.get("LAKEBASE_URL")
    if url:
        return url
    
    # Fall back to Databricks secrets
    try:
        from databricks.sdk import WorkspaceClient
        
        _w = WorkspaceClient()
        _SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
        _KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")
        
        secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
        return base64.b64decode(secret.value).decode("utf-8")
    except Exception as e:
        raise ValueError(
            "Could not find Lakebase credentials. Please either:\n"
            "1. Create a .env file with LAKEBASE_URL=postgresql://...\n"
            "2. Set LAKEBASE_URL environment variable\n"
            "3. Configure Databricks secret scope/key\n"
            f"Error: {e}"
        ) from e


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | list | None = None, executemany: bool = False) -> list[dict]:
    """
    Run a query against Lakebase and return rows as list[dict].
    
    Args:
        sql: SQL query string (can be SELECT, INSERT, UPDATE, DELETE)
        params: Parameters for the query (tuple, dict, or list of dicts for executemany)
        executemany: If True, execute the query multiple times with different parameters
        
    Returns:
        List of rows as dictionaries (empty list for non-SELECT queries)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if executemany and isinstance(params, list):
                cur.executemany(sql, params)
                conn.commit()
                return []
            else:
                cur.execute(sql, params)
                
                # For INSERT/UPDATE/DELETE, commit and return empty list
                if sql.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP')):
                    conn.commit()
                    return []
                
                # For SELECT, return rows
                return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
