"""Weather application package.

This package contains modules for fetching weather data from the NWS API
and syncing it to Lakebase Postgres database.
"""

from .lakebase import get_connection, get_engine, run_query, run_write
from .weather_client import WeatherFetcher, MAJOR_US_CITIES
from .sync_weather_to_db import sync_weather_data


def get_embedding_dim(model_name: str) -> int:
    """Get embedding dimensions for a given model name.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        int: Embedding dimensions for the model
        
    Raises:
        ValueError: If model is unknown
    """
    match model_name:
        case "sentence-transformers/all-MiniLM-L6-v2":
            return 384
        case "sentence-transformers/all-MiniLM-L12-v2":
            return 384
        case "sentence-transformers/all-mpnet-base-v2":
            return 768
        case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
            return 768
        case "BAAI/bge-small-en-v1.5":
            return 384
        case "BAAI/bge-base-en-v1.5":
            return 768
        case "BAAI/bge-large-en-v1.5":
            return 1024
        case "text-embedding-3-small":
            return 1536
        case "text-embedding-3-large":
            return 3072
        case _:
            raise ValueError(
                f"Unknown embedding model {model_name!r} - add its output "
                "dimension to get_embedding_dim() in weather_app/__init__.py"
            )


__all__ = [
    'get_connection',
    'get_engine', 
    'run_query',
    'run_write',
    'WeatherFetcher',
    'MAJOR_US_CITIES',
    'sync_weather_data',
    'get_embedding_dim',
]

__version__ = '1.0.0'
