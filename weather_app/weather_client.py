"""Weather.gov (NWS) API client for fetching alerts and forecasts.

Fetches weather alerts and forecasts from the National Weather Service API
and normalizes them into a unified document format suitable for RAG/embedding.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# NWS requires a User-Agent header
USER_AGENT = "DatabricksWeatherApp/1.0 (contact@example.com)"
NWS_BASE_URL = "https://api.weather.gov"

# Hardcoded coordinates for major US cities
MAJOR_US_CITIES = {
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Chicago, IL": (41.8781, -87.6298),
    "Houston, TX": (29.7604, -95.3698),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Philadelphia, PA": (39.9526, -75.1652),
    "San Antonio, TX": (29.4241, -98.4936),
    "San Diego, CA": (32.7157, -117.1611),
    "Dallas, TX": (32.7767, -96.7970),
    "Austin, TX": (30.2672, -97.7431),
    "San Francisco, CA": (37.7749, -122.4194),
    "Seattle, WA": (47.6062, -122.3321),
    "Denver, CO": (39.7392, -104.9903),
    "Boston, MA": (42.3601, -71.0589),
    "Miami, FL": (25.7617, -80.1918),
    "Atlanta, GA": (33.7490, -84.3880),
    "Portland, OR": (45.5152, -122.6784),
    "Las Vegas, NV": (36.1699, -115.1398),
    "Detroit, MI": (42.3314, -83.0458),
    "Nashville, TN": (36.1627, -86.7816),
}


class NWSAPIError(Exception):
    """Raised when NWS API returns an error."""
    pass


class GeocodeError(Exception):
    """Raised when geocoding fails."""
    pass


class WeatherFetcher:
    """Fetches weather data from NWS API and normalizes it into documents."""
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize the weather fetcher.
        
        Args:
            rate_limit_delay: Delay in seconds between API calls to avoid rate limiting
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = self._create_session()
        self._location_cache = {}  # Cache grid point lookups
        self._geocode_cache = {}  # Cache geocoding results
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic and proper headers."""
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        
        # Retry on transient failures
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _api_get(self, endpoint: str, base_url: str = NWS_BASE_URL) -> dict:
        """Make a GET request to an API with error handling.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            base_url: Base URL for the API (defaults to NWS)
            
        Returns:
            Response JSON as dict
            
        Raises:
            NWSAPIError: If API returns an error
        """
        url = f"{base_url}{endpoint}"
        logger.debug(f"Fetching: {url}")
        
        try:
            # Set Accept header for NWS API
            headers = {}
            if base_url == NWS_BASE_URL:
                headers["Accept"] = "application/geo+json"
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = f" - {error_data}"
                except:
                    error_detail = f" - {response.text[:200]}"
                
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code} for {url}{error_detail}"
                )
            
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            raise NWSAPIError(f"Request timeout for {endpoint}") from None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise NWSAPIError(f"Connection failed for {endpoint}: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"API error for {url}: {e}")
            raise NWSAPIError(f"Failed to fetch {endpoint}: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            raise NWSAPIError(f"Invalid JSON from {endpoint}: {e}") from e
    
    def _geocode_location(self, location_str: str) -> tuple[float, float]:
        """Look up coordinates for a city from the hardcoded dictionary.
        
        Args:
            location_str: Location string like "Chicago, IL" or "Austin, TX"
            
        Returns:
            Tuple of (lat, lon)
            
        Raises:
            GeocodeError: If city is not in the hardcoded list
        """
        # Check cache first
        if location_str in self._geocode_cache:
            logger.info(f"Using cached coordinates for '{location_str}'")
            return self._geocode_cache[location_str]
        
        # Normalize the location string
        location_str = location_str.strip()
        
        # Look up in hardcoded dictionary (exact match)
        if location_str in MAJOR_US_CITIES:
            coords = MAJOR_US_CITIES[location_str]
            self._geocode_cache[location_str] = coords
            logger.info(f"Found '{location_str}' at ({coords[0]:.4f}, {coords[1]:.4f})")
            return coords
        
        # Try case-insensitive match
        location_lower = location_str.lower()
        for city_name, coords in MAJOR_US_CITIES.items():
            if city_name.lower() == location_lower:
                self._geocode_cache[location_str] = coords
                logger.info(f"Found '{location_str}' (matched '{city_name}') at ({coords[0]:.4f}, {coords[1]:.4f})")
                return coords
        
        # City not found - provide helpful error message
        available_cities = ", ".join(sorted(MAJOR_US_CITIES.keys()))
        raise GeocodeError(
            f"City '{location_str}' not found in available cities. "
            f"Available cities: {available_cities}"
        )
    
    def _resolve_location(self, location: str | tuple[float, float]) -> dict:
        """Resolve a location to NWS grid point information.
        
        Args:
            location: Either "City, ST" string or (lat, lon) tuple
            
        Returns:
            Dict with keys: lat, lon, city, state, office, grid_x, grid_y
            
        Raises:
            NWSAPIError: If location cannot be resolved
            GeocodeError: If geocoding fails
        """
        cache_key = str(location)
        if cache_key in self._location_cache:
            logger.debug(f"Using cached location data for '{cache_key}'")
            return self._location_cache[cache_key]
        
        # Store original location for error messages
        original_location = location
        city_name = None
        state_name = None
        
        # Handle different input formats
        if isinstance(location, tuple) and len(location) == 2:
            lat, lon = location
            logger.info(f"Using provided coordinates: ({lat}, {lon})")
        elif isinstance(location, str):
            # Parse city/state from string if present
            if "," in location:
                parts = [p.strip() for p in location.split(",", 1)]
                if len(parts) == 2:
                    city_name = parts[0]
                    state_name = parts[1]
            
            # Geocode the location
            logger.info(f"Geocoding location: '{location}'")
            lat, lon = self._geocode_location(location)
        else:
            raise ValueError(
                f"Invalid location format: {location}. "
                f"Expected 'City, ST' string or (lat, lon) tuple."
            )
        
        # Get grid point from NWS
        try:
            logger.info(f"Fetching NWS grid point for ({lat:.4f}, {lon:.4f})...")
            point_data = self._api_get(f"/points/{lat:.4f},{lon:.4f}")
            
            if not point_data or "properties" not in point_data:
                raise NWSAPIError(
                    f"Invalid response from NWS points API for ({lat:.4f}, {lon:.4f})"
                )
            
            properties = point_data.get("properties", {})
            
            # Extract location info from NWS response
            rel_location = properties.get("relativeLocation", {}).get("properties", {})
            
            # Use parsed city/state if available, otherwise fall back to NWS data
            city = city_name or rel_location.get("city", "Unknown")
            state = state_name or rel_location.get("state", "Unknown")
            
            # Extract grid information
            office = properties.get("gridId")
            grid_x = properties.get("gridX")
            grid_y = properties.get("gridY")
            
            if not office or grid_x is None or grid_y is None:
                raise NWSAPIError(
                    f"Incomplete grid data from NWS for ({lat:.4f}, {lon:.4f}). "
                    f"Got office={office}, grid_x={grid_x}, grid_y={grid_y}"
                )
            
            result = {
                "lat": lat,
                "lon": lon,
                "city": city,
                "state": state,
                "office": office,
                "grid_x": grid_x,
                "grid_y": grid_y,
            }
            
            logger.info(
                f"Resolved '{original_location}' to {city}, {state} "
                f"(grid: {office}/{grid_x},{grid_y})"
            )
            
            self._location_cache[cache_key] = result
            return result
            
        except NWSAPIError as e:
            logger.error(f"Failed to resolve NWS grid for '{original_location}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error resolving location '{original_location}': {e}")
            raise NWSAPIError(f"Failed to resolve location '{original_location}': {e}") from e
    
    def _fetch_alerts_for_state(self, state: str) -> list[dict]:
        """Fetch active alerts for a given state.
        
        Args:
            state: Two-letter state code (e.g., 'CA', 'TX')
            
        Returns:
            List of alert feature dicts
        """
        try:
            data = self._api_get(f"/alerts/active?area={state}")
            return data.get("features", [])
        except NWSAPIError as e:
            logger.warning(f"Failed to fetch alerts for {state}: {e}")
            return []
    
    def _fetch_forecast(self, office: str, grid_x: int, grid_y: int) -> list[dict]:
        """Fetch forecast for a grid point.
        
        Args:
            office: NWS office code (e.g., 'LOX')
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            
        Returns:
            List of forecast period dicts
        """
        try:
            data = self._api_get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")
            return data.get("properties", {}).get("periods", [])
        except NWSAPIError as e:
            logger.warning(f"Failed to fetch forecast for {office}/{grid_x},{grid_y}: {e}")
            return []
    
    def _fetch_hourly_forecast(self, office: str, grid_x: int, grid_y: int) -> list[dict]:
        """Fetch hourly forecast for a grid point.
        
        Args:
            office: NWS office code
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            
        Returns:
            List of hourly forecast period dicts
        """
        try:
            data = self._api_get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly")
            return data.get("properties", {}).get("periods", [])
        except NWSAPIError as e:
            logger.warning(f"Failed to fetch hourly forecast for {office}/{grid_x},{grid_y}: {e}")
            return []
    
    def _normalize_alert(self, alert: dict, location_info: dict) -> dict:
        """Normalize an NWS alert into a document record.
        
        Args:
            alert: Alert feature from NWS API
            location_info: Location metadata dict
            
        Returns:
            Normalized document dict
        """
        props = alert.get("properties", {})
        
        # Build narrative from description + instruction
        narrative_parts = []
        if props.get("description"):
            narrative_parts.append(props["description"])
        if props.get("instruction"):
            narrative_parts.append(f"Instructions: {props['instruction']}")
        narrative_text = "\n\n".join(narrative_parts) or props.get("headline", "")
        
        return {
            "id": alert.get("id", ""),  # NWS provides unique IDs
            "location": f"{location_info['city']}, {location_info['state']}",
            "location_lat": location_info["lat"],
            "location_lon": location_info["lon"],
            "source_type": "alert",
            "headline": props.get("headline", ""),
            "event": props.get("event", ""),
            "severity": props.get("severity", ""),
            "narrative_text": narrative_text,
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective"),
            "expires_at": props.get("expires"),
            "payload": alert,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _normalize_forecast(self, 
                           period: dict, 
                           location_info: dict, 
                           source_type: Literal["forecast", "hourly_forecast"]) -> dict:
        """Normalize a forecast period into a document record.
        
        Args:
            period: Forecast period from NWS API
            location_info: Location metadata dict
            source_type: Either 'forecast' or 'hourly_forecast'
            
        Returns:
            Normalized document dict
        """
        # Generate stable ID from location + time + period
        id_components = (
            f"{location_info['lat']},{location_info['lon']}"
            f"-{period.get('startTime', '')}"
            f"-{period.get('name', '')}"
        )
        doc_id = hashlib.sha256(id_components.encode()).hexdigest()[:16]
        
        # Use detailedForecast for narrative
        narrative_text = period.get("detailedForecast") or period.get("shortForecast", "")
        
        return {
            "id": doc_id,
            "location": f"{location_info['city']}, {location_info['state']}",
            "location_lat": location_info["lat"],
            "location_lon": location_info["lon"],
            "source_type": source_type,
            "headline": period.get("name", ""),
            "event": period.get("shortForecast", ""),
            "severity": None,
            "narrative_text": narrative_text,
            "issued_at": period.get("startTime"),
            "effective_at": period.get("startTime"),
            "expires_at": period.get("endTime"),
            "payload": period,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def fetch_weather_documents(self, config: dict) -> dict[str, Any]:
        """Fetch weather documents for given locations.
        
        Args:
            config: Dict with keys:
                - locations: List of "City, ST" strings or (lat, lon) tuples
                - limit: Max number of documents to return PER CITY (default: 50)
                - include_alerts: Whether to fetch alerts (default: True)
                - include_forecasts: Whether to fetch forecasts (default: True)
                - include_hourly: Whether to fetch hourly forecasts (default: False)
                
        Returns:
            Dict with keys:
                - documents: List of normalized document dicts
                - stats: Summary statistics
                - errors: List of error messages encountered
        """
        locations = config.get("locations", [])
        limit = config.get("limit", 50)
        include_alerts = config.get("include_alerts", True)
        include_forecasts = config.get("include_forecasts", True)
        include_hourly = config.get("include_hourly", False)
        
        documents = []
        errors = []
        stats = {
            "locations_processed": 0,
            "alerts_fetched": 0,
            "forecasts_fetched": 0,
            "hourly_forecasts_fetched": 0,
        }
        
        # Track states for batch alert fetching
        states_to_fetch = set()
        location_infos = []
        
        # Step 1: Resolve all locations
        if not locations:
            logger.warning("No locations provided in config")
            return {
                "documents": [],
                "stats": stats,
                "errors": ["No locations provided"],
            }
        
        logger.info(f"Resolving {len(locations)} location(s)...")
        for i, loc in enumerate(locations, 1):
            try:
                logger.info(f"[{i}/{len(locations)}] Processing location: {loc}")
                loc_info = self._resolve_location(loc)
                location_infos.append(loc_info)
                states_to_fetch.add(loc_info["state"])
                stats["locations_processed"] += 1
                logger.info(f"✓ Successfully resolved: {loc_info['city']}, {loc_info['state']}")
            except (NWSAPIError, GeocodeError, ValueError) as e:
                error_msg = f"Failed to resolve '{loc}': {str(e)}"
                errors.append(error_msg)
                logger.warning(f"✗ {error_msg}")
                logger.exception(f"Full error details for '{loc}':")
        
        # Step 2: Fetch alerts by state (batch operation)
        state_alerts = {}
        if include_alerts:
            logger.info(f"Fetching alerts for {len(states_to_fetch)} states...")
            for state in states_to_fetch:
                alerts = self._fetch_alerts_for_state(state)
                state_alerts[state] = alerts
                stats["alerts_fetched"] += len(alerts)
        
        # Step 3: Process each location
        logger.info(f"Processing {len(location_infos)} locations...")
        for loc_info in location_infos:
            # Track documents added for this specific location
            location_doc_count = 0
            
            # Add relevant alerts for this location
            if include_alerts and loc_info["state"] in state_alerts:
                for alert in state_alerts[loc_info["state"]]:
                    if location_doc_count >= limit:
                        break
                    doc = self._normalize_alert(alert, loc_info)
                    documents.append(doc)
                    location_doc_count += 1
            
            # Fetch and add forecasts
            if include_forecasts and location_doc_count < limit:
                try:
                    periods = self._fetch_forecast(
                        loc_info["office"],
                        loc_info["grid_x"],
                        loc_info["grid_y"]
                    )
                    for period in periods:
                        if location_doc_count >= limit:
                            break
                        doc = self._normalize_forecast(period, loc_info, "forecast")
                        documents.append(doc)
                        location_doc_count += 1
                    stats["forecasts_fetched"] += len(periods)
                except Exception as e:
                    errors.append(f"Failed to fetch forecast for {loc_info['city']}: {e}")
            
            # Fetch and add hourly forecasts
            if include_hourly and location_doc_count < limit:
                try:
                    periods = self._fetch_hourly_forecast(
                        loc_info["office"],
                        loc_info["grid_x"],
                        loc_info["grid_y"]
                    )
                    for period in periods:
                        if location_doc_count >= limit:
                            break
                        doc = self._normalize_forecast(period, loc_info, "hourly_forecast")
                        documents.append(doc)
                        location_doc_count += 1
                    stats["hourly_forecasts_fetched"] += len(periods)
                except Exception as e:
                    errors.append(f"Failed to fetch hourly forecast for {loc_info['city']}: {e}")
            
            logger.info(f"Added {location_doc_count} documents for {loc_info['city']}, {loc_info['state']}")
        
        logger.info(f"Fetched {len(documents)} total documents")
        
        return {
            "documents": documents,  # No need to slice - limit is now per-city
            "stats": stats,
            "errors": errors,
        }
