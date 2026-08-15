#!/usr/bin/env python3
"""
Weather Data Dashboard - Flask Application

A web dashboard for viewing and syncing weather data from the National Weather Service API.
Displays the latest weather documents for all tracked cities with real-time sync capability.
"""

from flask import Flask, render_template_string, jsonify, request
import yaml
import os
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Add the current directory to the path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weather_app import lakebase
from weather_app.sync_weather_to_db import sync_weather_data

app = Flask(__name__)

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'app.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Embedding model loading (singleton)
EMBEDDING_MODEL = None

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        model_name = config['embeddings']['model_name']
        EMBEDDING_MODEL = SentenceTransformer(model_name, cache_folder="/tmp/.cache/huggingface")
    return EMBEDDING_MODEL

# HTML template for the main page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin: 0 0 30px 0;
            font-size: 32px;
        }
        .controls {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            color: white;
        }
        .sync-form {
            max-width: 800px;
            margin: 0 auto;
        }
        .form-section {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .form-section h3 {
            margin: 0 0 15px 0;
            font-size: 16px;
            font-weight: 600;
        }
        .cities-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        .city-checkbox {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .city-checkbox:hover {
            background: rgba(255,255,255,0.2);
        }
        .city-checkbox input[type="checkbox"] {
            cursor: pointer;
            width: 16px;
            height: 16px;
        }
        .city-checkbox label {
            cursor: pointer;
            font-size: 14px;
            user-select: none;
        }
        .select-all {
            margin-bottom: 10px;
            padding: 10px;
            background: rgba(255,255,255,0.15);
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .select-all input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .select-all label {
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }
        .limit-input {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .limit-input label {
            font-weight: 600;
            font-size: 14px;
        }
        .limit-input input {
            padding: 8px 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 14px;
            width: 100px;
        }
        .limit-input input:focus {
            outline: none;
            border-color: rgba(255,255,255,0.6);
            background: rgba(255,255,255,0.2);
        }
        .form-actions {
            text-align: center;
            margin-top: 20px;
        }
        .sync-btn {
            background-color: white;
            color: #667eea;
            padding: 14px 40px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .sync-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .sync-btn:disabled {
            background-color: #cccccc;
            color: #666;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            margin-top: 15px;
            padding: 12px 20px;
            border-radius: 6px;
            display: none;
            font-weight: 500;
        }
        .status.success {
            background-color: rgba(255,255,255,0.2);
            color: white;
            display: block;
        }
        .status.error {
            background-color: rgba(255,59,48,0.9);
            color: white;
            display: block;
        }
        .status.loading {
            background-color: rgba(255,255,255,0.2);
            color: white;
            display: block;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 8px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
        }
        .stat-card p {
            margin: 0 0 8px 0;
            font-size: 32px;
            font-weight: bold;
        }
        .stat-card .subtitle {
            font-size: 11px;
            opacity: 0.85;
            margin: 0;
            font-weight: normal;
        }
        .city-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
        }
        @media (max-width: 1200px) {
            .city-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        .city-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid #e9ecef;
        }
        .city-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .city-name {
            font-size: 22px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .document {
            margin-bottom: 12px;
            padding: 14px;
            background-color: white;
            border-left: 4px solid #4CAF50;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .document.alert {
            border-left-color: #ff5252;
            background-color: #fff5f5;
        }
        .document.forecast {
            border-left-color: #2196F3;
            background-color: #f0f8ff;
        }
        .document.hourly_forecast {
            border-left-color: #4CAF50;
            background-color: #f1f8f4;
        }
        .document:hover {
            border-left-width: 6px;
            padding-left: 12px;
        }
        .doc-type {
            font-weight: 600;
            color: #4CAF50;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .doc-type.alert {
            color: #ff5252;
        }
        .doc-type.forecast {
            color: #2196F3;
        }
        .doc-type.hourly_forecast {
            color: #4CAF50;
        }
        .doc-severity {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 600;
            margin-left: 8px;
            text-transform: uppercase;
        }
        .severity-severe { background: #ff5252; color: white; }
        .severity-moderate { background: #ff9800; color: white; }
        .severity-minor { background: #ffc107; color: #333; }
        .doc-event {
            font-size: 12px;
            color: #666;
            margin-top: 3px;
        }
        .doc-headline {
            margin: 5px 0;
            color: #333;
            line-height: 1.4;
            font-size: 14px;
        }
        .doc-time {
            font-size: 11px;
            color: #999;
            margin-top: 5px;
        }
        .no-data {
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            color: #666;
            font-size: 14px;
        }
        .search-section {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            color: white;
        }
        .search-section h2 {
            margin: 0 0 15px 0;
            font-size: 20px;
            font-weight: 600;
        }
        .search-form {
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }
        .search-input-group {
            flex: 1;
        }
        .search-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 6px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 15px;
            transition: all 0.2s;
        }
        .search-input::placeholder {
            color: rgba(255,255,255,0.7);
        }
        .search-input:focus {
            outline: none;
            border-color: white;
            background: rgba(255,255,255,0.2);
        }
        .search-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .top-k-input {
            width: 80px;
            padding: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 6px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 14px;
            text-align: center;
        }
        .top-k-input:focus {
            outline: none;
            border-color: white;
            background: rgba(255,255,255,0.2);
        }
        .search-btn {
            background-color: white;
            color: #11998e;
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            white-space: nowrap;
        }
        .search-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .search-btn:disabled {
            background-color: #cccccc;
            color: #666;
            cursor: not-allowed;
            transform: none;
        }
        .search-results {
            margin-top: 20px;
            display: none;
        }
        .search-results.active {
            display: block;
        }
        .search-result-header {
            background: rgba(255,255,255,0.2);
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 14px;
        }
        .search-result-item {
            background: rgba(255,255,255,0.15);
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }
        .search-result-item:hover {
            background: rgba(255,255,255,0.25);
            transform: translateX(5px);
        }
        .result-location {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .result-type {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            background: rgba(255,255,255,0.3);
            text-transform: uppercase;
        }
        .result-headline {
            font-size: 14px;
            margin-bottom: 8px;
            font-weight: 500;
        }
        .result-text {
            font-size: 13px;
            line-height: 1.5;
            opacity: 0.95;
            margin-bottom: 8px;
        }
        .result-similarity {
            font-size: 12px;
            opacity: 0.8;
            font-family: monospace;
        }
        .search-error {
            background: rgba(255,59,48,0.9);
            padding: 12px 16px;
            border-radius: 6px;
            margin-top: 15px;
            display: none;
        }
        .search-error.active {
            display: block;
        }
        @media (max-width: 768px) {
            .city-grid {
                grid-template-columns: 1fr;
            }
            .stats {
                grid-template-columns: 1fr;
            }
        }
        .view-more {
            text-align: center;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e9ecef;
        }
        .view-more a {
            color: #667eea;
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            transition: color 0.2s;
        }
        .view-more a:hover {
            color: #764ba2;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ {{ title }}</h1>
        
        <div class="search-section">
            <h2>🔍 Semantic Weather Search</h2>
            <form id="searchForm" class="search-form" onsubmit="performSemanticSearch(event)">
                <div class="search-input-group">
                    <input id="searchQuery" class="search-input" type="text"
                           placeholder="E.g., risk of flooding near rivers, severe thunderstorm warning" required>
                </div>
                <div class="search-controls">
                    <input id="searchTopK" class="top-k-input" type="number" value="5" min="1" max="20" title="Results (1-20)">
                    <button type="submit" class="search-btn" id="searchBtn">🔎 Search</button>
                </div>
            </form>
            <div class="search-error" id="searchError"></div>
            <div class="search-results" id="searchResults">
                <div class="search-result-header" id="searchResultHeader"></div>
                <div id="searchResultsList"></div>
            </div>
        </div>
        
        <div class="controls">
            <form class="sync-form" id="syncForm" onsubmit="syncData(event)">
                <div class="form-section">
                    <h3>📍 Select Cities to Sync</h3>
                    <div class="select-all">
                        <input type="checkbox" id="selectAll" onchange="toggleAllCities()" checked>
                        <label for="selectAll">Select All Cities</label>
                    </div>
                    <div class="cities-grid">
                        {% for city_data in cities %}
                        <div class="city-checkbox">
                            <input type="checkbox" id="city-{{ loop.index }}" name="city" value="{{ city_data.city }}" checked>
                            <label for="city-{{ loop.index }}">{{ city_data.city }}</label>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="form-section">
                    <div class="limit-input">
                        <label for="limit">📊 Documents per city:</label>
                        <input type="number" id="limit" name="limit" value="50" min="1" max="1000">
                    </div>
                </div>
                
                <div class="form-actions">
                    <button type="submit" class="sync-btn" id="syncBtn">🔄 Sync Weather Data</button>
                </div>
            </form>
            <div class="status" id="status"></div>
        </div>
        
        <div class="stats">
            <div class="stat-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                <h3>🚨 Active Alerts</h3>
                <p>{{ stats.active_alerts }}</p>
                <div class="subtitle">Currently in effect</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4e54c8 0%, #8f94fb 100%);">
                <h3>📦 Total Documents</h3>
                <p>{{ "{:,}".format(stats.total_documents) }}</p>
                <div class="subtitle">Across all cities</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%);">
                <h3>🌤️ Latest Forecasts</h3>
                <p>{{ stats.today_forecasts }}</p>
                <div class="subtitle">Issued today</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <h3>📍 Cities Tracked</h3>
                <p style="font-size: 28px;">{{ stats.cities_ratio }}</p>
                <div class="subtitle">With today's data</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>🔄 Data Freshness</h3>
                <p style="font-size: 20px;">{{ stats.last_sync_text }}</p>
                <div class="subtitle">Last sync time</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <h3>📈 This Week</h3>
                <p>{{ "{:,}".format(stats.week_documents) }}</p>
                <div class="subtitle">Total documents</div>
            </div>
        </div>
        
        <div class="city-grid">
            {% for city_data in cities %}
            <div class="city-card">
                <div class="city-name">
                    <span>📍</span>
                    <span>{{ city_data.city }}</span>
                </div>
                {% if city_data.documents %}
                    {% for doc in city_data.documents[:5] %}
                    <div class="document {{ doc.source_type }}">
                        <div class="doc-type {{ doc.source_type }}">
                            {{ doc.source_type }}
                            {% if doc.severity %}
                            <span class="doc-severity severity-{{ doc.severity.lower() }}">{{ doc.severity }}</span>
                            {% endif %}
                        </div>
                        {% if doc.event %}
                        <div class="doc-event">⚠️ {{ doc.event }}</div>
                        {% endif %}
                        <div class="doc-headline">{{ doc.headline }}</div>
                        <div class="doc-time">{{ doc.issued_at }}</div>
                    </div>
                    {% endfor %}
                    {% if city_data.documents|length > 5 %}
                    <div class="view-more">
                        <a href="/city/{{ city_data.city|urlencode }}">📄 View {{ city_data.documents|length - 5 }} more →</a>
                    </div>
                    {% endif %}
                {% else %}
                    <div class="no-data">No data available</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="footer">
            <p>Data sourced from National Weather Service API</p>
            <p>Powered by Databricks Lakebase</p>
        </div>
    </div>
    
    <script>
        function toggleAllCities() {
            const selectAll = document.getElementById('selectAll');
            const cityCheckboxes = document.querySelectorAll('input[name="city"]');
            cityCheckboxes.forEach(checkbox => {
                checkbox.checked = selectAll.checked;
            });
        }
        
        function syncData(event) {
            event.preventDefault();
            
            const btn = document.getElementById('syncBtn');
            const status = document.getElementById('status');
            const form = document.getElementById('syncForm');
            
            // Get selected cities
            const selectedCities = Array.from(document.querySelectorAll('input[name="city"]:checked'))
                .map(checkbox => checkbox.value);
            
            // Get limit
            const limit = parseInt(document.getElementById('limit').value);
            
            // Validate
            if (selectedCities.length === 0) {
                status.className = 'status error';
                status.textContent = '✗ Please select at least one city';
                return;
            }
            
            btn.disabled = true;
            status.className = 'status loading';
            status.textContent = `⏳ Syncing ${selectedCities.length} ${selectedCities.length === 1 ? 'city' : 'cities'} from NWS API... This may take a minute.`;
            
            fetch('/weather/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    locations: selectedCities,
                    limit: limit
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        status.className = 'status success';
                        let message = `✓ Successfully synced! Fetched: ${data.documents_fetched}, Upserted: ${data.documents_upserted}`;
                        
                        // Add embedding info if available
                        if (data.chunks_embedded && data.chunks_embedded > 0) {
                            message += `, Embeddings: ${data.chunks_embedded} chunks`;
                        }
                        
                        // Add per-city breakdown if available
                        if (data.city_breakdown && data.city_breakdown.length > 0) {
                            message += '<br><br>📍 <strong>Documents per City:</strong><br>';
                            data.city_breakdown.forEach(city => {
                                message += `&nbsp;&nbsp;${city.location}: ${city.count} docs (${city.types} types)<br>`;
                            });
                        }
                        
                        status.innerHTML = message;
                        setTimeout(() => location.reload(), 3000);
                    } else {
                        status.className = 'status error';
                        status.textContent = `✗ Sync failed: ${data.error}`;
                        btn.disabled = false;
                    }
                })
                .catch(error => {
                    status.className = 'status error';
                    status.textContent = `✗ Error: ${error.message}`;
                    btn.disabled = false;
                });
        }
        
        function performSemanticSearch(event) {
            event.preventDefault();
            
            const btn = document.getElementById('searchBtn');
            const errorDiv = document.getElementById('searchError');
            const resultsDiv = document.getElementById('searchResults');
            const resultsHeader = document.getElementById('searchResultHeader');
            const resultsList = document.getElementById('searchResultsList');
            const queryInput = document.getElementById('searchQuery');
            const topKInput = document.getElementById('searchTopK');
            
            // Get query and top_k
            const query = queryInput.value.trim();
            const topK = parseInt(topKInput.value) || 5;
            
            // Reset UI
            errorDiv.classList.remove('active');
            resultsDiv.classList.remove('active');
            
            // Validate
            if (!query) {
                errorDiv.textContent = '✗ Please enter a search query';
                errorDiv.classList.add('active');
                return;
            }
            
            // Disable button and show loading state
            btn.disabled = true;
            btn.textContent = '🔍 Searching...';
            
            // Call backend
            fetch('/weather/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: topK
                })
            })
                .then(response => response.json())
                .then(data => {
                    btn.disabled = false;
                    btn.textContent = '🔎 Search';
                    
                    if (data.success) {
                        // Display results
                        if (data.results && data.results.length > 0) {
                            resultsHeader.textContent = `Found ${data.results.length} matching weather documents:`;
                            
                            resultsList.innerHTML = data.results.map(result => `
                                <div class="search-result-item">
                                    <div class="result-location">
                                        📍 ${result.location}
                                        <span class="result-type">${result.source_type}</span>
                                    </div>
                                    ${result.headline ? `<div class="result-headline"><strong>${result.headline}</strong></div>` : ''}
                                    <div class="result-text">${result.chunk_text}</div>
                                    <div class="result-similarity">Similarity: ${(result.similarity * 100).toFixed(1)}%</div>
                                </div>
                            `).join('');
                            
                            resultsDiv.classList.add('active');
                        } else {
                            // No results or message from backend
                            resultsHeader.textContent = data.message || 'No matching documents found.';
                            resultsList.innerHTML = '';
                            resultsDiv.classList.add('active');
                        }
                    } else {
                        // Error from backend
                        errorDiv.textContent = `✗ Search failed: ${data.error}`;
                        errorDiv.classList.add('active');
                    }
                })
                .catch(error => {
                    btn.disabled = false;
                    btn.textContent = '🔎 Search';
                    errorDiv.textContent = `✗ Error: ${error.message}`;
                    errorDiv.classList.add('active');
                });
        }
        
        // Auto-refresh every 5 minutes
        setTimeout(() => location.reload(), 300000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Display today's forecasts and alerts for each city"""
    try:
        # Get latest forecasts and alerts for all cities (today or earlier)
        # Optimized to use composite index (source_type, issued_at, location)
        query = f'''
            SELECT 
                location,
                source_type,
                headline,
                issued_at,
                severity,
                event
            FROM "{config['database']['schema']}".{config['database']['table']}
            WHERE 
                source_type IN ('forecast', 'alert', 'hourly_forecast')
                AND issued_at >= CURRENT_DATE
            ORDER BY 
                location,
                issued_at ASC,
                CASE 
                    WHEN source_type = 'alert' THEN 1
                    WHEN source_type = 'forecast' THEN 2
                    ELSE 3
                END
        '''
        
        results = lakebase.run_query(query)
        
        # Get comprehensive statistics
        stats_query = f'''
            WITH today_data AS (
                SELECT * FROM "{config['database']['schema']}".{config['database']['table']}
                WHERE DATE(issued_at) = CURRENT_DATE
            ),
            week_data AS (
                SELECT COUNT(*) as week_count
                FROM "{config['database']['schema']}".{config['database']['table']}
                WHERE DATE(issued_at) >= CURRENT_DATE - INTERVAL '7 days'
            ),
            total_data AS (
                SELECT 
                    COUNT(*) as total_docs,
                    COUNT(DISTINCT location) as total_cities,
                    MAX(synced_at) as last_sync
                FROM "{config['database']['schema']}".{config['database']['table']}
            )
            SELECT 
                -- Active alerts issued today (currently in effect)
                (SELECT COUNT(*) FROM today_data 
                 WHERE source_type = 'alert') as active_alerts,
                
                -- Total documents across all time
                (SELECT total_docs FROM total_data) as total_documents,
                
                -- Latest forecasts issued today
                (SELECT COUNT(*) FROM today_data 
                 WHERE source_type = 'forecast') as today_forecasts,
                
                -- Cities with data today
                (SELECT COUNT(DISTINCT location) FROM today_data) as cities_today,
                
                -- Total cities ever tracked
                (SELECT total_cities FROM total_data) as cities_total,
                
                -- Last sync time
                (SELECT last_sync FROM total_data) as last_sync,
                
                -- Documents from this week
                (SELECT week_count FROM week_data) as week_documents
        '''
        stats_result = lakebase.run_query(stats_query)
        stats = stats_result[0] if stats_result else {}
        
        # Get all configured cities (ensure all 10 cities are shown)
        all_cities = config['weather']['locations']
        
        # Organize data by city
        cities_data = {city: [] for city in all_cities}  # Initialize all cities
        for row in results:
            city = row['location']
            if city not in cities_data:
                cities_data[city] = []
            cities_data[city].append({
                'source_type': row['source_type'],
                'headline': row['headline'],
                'issued_at': row['issued_at'].strftime('%Y-%m-%d %H:%M UTC') if row['issued_at'] else 'N/A',
                'severity': row.get('severity', ''),
                'event': row.get('event', '')
            })
        
        # Convert to list for template (sorted by city name)
        cities_list = [{'city': city, 'documents': docs} for city, docs in sorted(cities_data.items())]
        
        # Calculate time since last sync
        last_sync_text = 'Never'
        if stats.get('last_sync'):
            from datetime import timedelta
            time_diff = datetime.now() - stats['last_sync'].replace(tzinfo=None)
            if time_diff < timedelta(minutes=1):
                last_sync_text = 'Just now'
            elif time_diff < timedelta(hours=1):
                mins = int(time_diff.total_seconds() / 60)
                last_sync_text = f'{mins} min ago' if mins == 1 else f'{mins} mins ago'
            elif time_diff < timedelta(days=1):
                hours = int(time_diff.total_seconds() / 3600)
                last_sync_text = f'{hours} hr ago' if hours == 1 else f'{hours} hrs ago'
            else:
                days = time_diff.days
                last_sync_text = f'{days} day ago' if days == 1 else f'{days} days ago'
        
        # Format stats
        total_cities = len(config['weather']['locations'])
        stats_formatted = {
            'active_alerts': stats.get('active_alerts', 0),
            'total_documents': stats.get('total_documents', 0),
            'today_forecasts': stats.get('today_forecasts', 0),
            'cities_today': stats.get('cities_today', 0),
            'cities_total': total_cities,
            'cities_ratio': f"{stats.get('cities_today', 0)}/{total_cities}",
            'last_sync_text': last_sync_text,
            'week_documents': stats.get('week_documents', 0),
            'today': datetime.now().strftime('%B %d, %Y')
        }
        
        return render_template_string(
            HTML_TEMPLATE, 
            cities=cities_list, 
            stats=stats_formatted,
            title=config['dashboard']['title']
        )
        
    except Exception as e:
        return f"<h1>Error loading data</h1><p>{str(e)}</p>", 500

@app.route('/city/<path:city_name>')
def city_detail(city_name):
    """Display all documents for a specific city"""
    try:
        from urllib.parse import unquote
        city_name = unquote(city_name)
        
        # Get all documents for this city, grouped by date
        query = f'''
            SELECT 
                location,
                source_type,
                headline,
                issued_at,
                severity,
                event,
                payload,
                DATE(issued_at) as issue_date
            FROM "{config['database']['schema']}".{config['database']['table']}
            WHERE 
                location = '{city_name}'
                AND source_type IN ('forecast', 'alert', 'hourly_forecast')
                AND DATE(issued_at) >= CURRENT_DATE
            ORDER BY 
                issue_date ASC,
                issued_at ASC
        '''
        
        results = lakebase.run_query(query)
        
        # Group documents by date and source type
        from collections import defaultdict
        import json
        docs_by_date = defaultdict(lambda: {'alert': [], 'forecast': [], 'hourly_forecast': []})
        
        for row in results:
            date_str = row['issue_date'].strftime('%Y-%m-%d')
            
            # Parse payload for forecast data
            forecast_data = {}
            if row.get('payload'):
                try:
                    payload = json.loads(row['payload']) if isinstance(row['payload'], str) else row['payload']
                    forecast_data = {
                        'temperature': payload.get('temperature'),
                        'temperatureUnit': payload.get('temperatureUnit', 'F'),
                        'shortForecast': payload.get('shortForecast', ''),
                        'windSpeed': payload.get('windSpeed', ''),
                        'windDirection': payload.get('windDirection', ''),
                        'precipitation': payload.get('probabilityOfPrecipitation', {}).get('value', 0) if payload.get('probabilityOfPrecipitation') else 0,
                        'detailedForecast': payload.get('detailedForecast', ''),
                        'isDaytime': payload.get('isDaytime', True)
                    }
                except:
                    pass
            
            doc = {
                'headline': row['headline'],
                'issued_at': row['issued_at'].strftime('%H:%M UTC') if row['issued_at'] else 'N/A',
                'severity': row.get('severity', ''),
                'event': row.get('event', ''),
                **forecast_data
            }
            docs_by_date[date_str][row['source_type']].append(doc)
        
        # Calculate total count
        total_count = sum(len(docs['alert']) + len(docs['forecast']) + len(docs['hourly_forecast']) 
                         for docs in docs_by_date.values())
        
        # Simple HTML for city detail page
        city_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{city_name} - Weather Data</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #667eea;
                    margin-bottom: 10px;
                }}
                .back-link {{
                    display: inline-block;
                    margin-bottom: 20px;
                    color: #667eea;
                    text-decoration: none;
                    font-weight: 600;
                }}
                .back-link:hover {{
                    text-decoration: underline;
                }}
                .count {{
                    color: #666;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
                .date-section {{
                    margin-bottom: 30px;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .date-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px 20px;
                    font-weight: 600;
                    font-size: 16px;
                }}
                .columns-container {{
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 0;
                    border-top: 1px solid #e0e0e0;
                }}
                .column {{
                    padding: 15px;
                    border-right: 1px solid #e0e0e0;
                    min-height: 100px;
                }}
                .column:last-child {{
                    border-right: none;
                }}
                .column-header {{
                    font-weight: 600;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 12px;
                    padding-bottom: 8px;
                    border-bottom: 2px solid;
                }}
                .column.alerts .column-header {{
                    color: #ff5252;
                    border-bottom-color: #ff5252;
                }}
                .column.forecasts .column-header {{
                    color: #2196F3;
                    border-bottom-color: #2196F3;
                }}
                .column.hourly .column-header {{
                    color: #4CAF50;
                    border-bottom-color: #4CAF50;
                }}
                .doc-item {{
                    margin-bottom: 12px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    font-size: 13px;
                }}
                .column.alerts .doc-item {{
                    background-color: #fff5f5;
                    border-left: 3px solid #ff5252;
                }}
                .column.forecasts .doc-item {{
                    background-color: #f0f8ff;
                    border-left: 3px solid #2196F3;
                }}
                .column.hourly .doc-item {{
                    background-color: #f1f8f4;
                    border-left: 3px solid #4CAF50;
                }}
                .doc-headline {{
                    margin-bottom: 6px;
                    color: #333;
                    line-height: 1.4;
                }}
                .doc-time {{
                    font-size: 11px;
                    color: #999;
                }}
                .forecast-temp {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 4px;
                }}
                .forecast-condition {{
                    font-size: 13px;
                    color: #666;
                    margin-bottom: 6px;
                }}
                .forecast-details {{
                    font-size: 11px;
                    color: #999;
                    margin-top: 4px;
                }}
                .forecast-details span {{
                    margin-right: 8px;
                }}
                .doc-severity {{
                    display: inline-block;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 9px;
                    font-weight: 600;
                    margin-bottom: 4px;
                    text-transform: uppercase;
                }}
                .severity-severe {{ background: #ff5252; color: white; }}
                .severity-moderate {{ background: #ff9800; color: white; }}
                .severity-minor {{ background: #ffc107; color: #333; }}
                .doc-event {{
                    font-size: 11px;
                    color: #666;
                    margin-bottom: 4px;
                }}
                .empty-column {{
                    color: #999;
                    font-style: italic;
                    font-size: 12px;
                }}
                .temp-toggle {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 20px;
                    padding: 12px 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px;
                    color: white;
                    font-weight: 600;
                }}
                .temp-toggle label {{
                    cursor: pointer;
                    user-select: none;
                }}
                .toggle-switch {{
                    position: relative;
                    display: inline-block;
                    width: 50px;
                    height: 24px;
                }}
                .toggle-switch input {{
                    opacity: 0;
                    width: 0;
                    height: 0;
                }}
                .toggle-slider {{
                    position: absolute;
                    cursor: pointer;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: rgba(255,255,255,0.3);
                    transition: 0.3s;
                    border-radius: 24px;
                }}
                .toggle-slider:before {{
                    position: absolute;
                    content: "";
                    height: 18px;
                    width: 18px;
                    left: 3px;
                    bottom: 3px;
                    background-color: white;
                    transition: 0.3s;
                    border-radius: 50%;
                }}
                input:checked + .toggle-slider {{
                    background-color: rgba(255,255,255,0.5);
                }}
                input:checked + .toggle-slider:before {{
                    transform: translateX(26px);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-link">← Back to Dashboard</a>
                <h1>📍 {city_name}</h1>
                <div class="count">{total_count} document(s) found across {len(docs_by_date)} date(s)</div>
                
                <div class="temp-toggle">
                    <label for="tempUnit">🌡️ Temperature Unit:</label>
                    <span id="fahrenheitLabel" style="opacity: 1;">°F</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="tempUnit">
                        <span class="toggle-slider"></span>
                    </label>
                    <span id="celsiusLabel" style="opacity: 0.5;">°C</span>
                </div>
        '''
        
        if docs_by_date:
            for date_str in sorted(docs_by_date.keys(), reverse=False):
                docs = docs_by_date[date_str]
                city_html += f'''
                <div class="date-section">
                    <div class="date-header">📅 {date_str}</div>
                    <div class="columns-container">
                        <div class="column alerts">
                            <div class="column-header">🚨 Alerts ({len(docs['alert'])})</div>
                '''
                
                if docs['alert']:
                    for doc in docs['alert']:
                        severity_badge = f'<div class="doc-severity severity-{doc["severity"].lower()}">{doc["severity"]}</div>' if doc['severity'] else ''
                        event_line = f'<div class="doc-event">⚠️ {doc["event"]}</div>' if doc['event'] else ''
                        city_html += f'''
                            <div class="doc-item">
                                {severity_badge}
                                {event_line}
                                <div class="doc-headline">{doc['headline']}</div>
                                <div class="doc-time">{doc['issued_at']}</div>
                            </div>
                        '''
                else:
                    city_html += '<div class="empty-column">No alerts</div>'
                
                city_html += f'''
                        </div>
                        <div class="column forecasts">
                            <div class="column-header">🌤️ Forecasts ({len(docs['forecast'])})</div>
                '''
                
                if docs['forecast']:
                    for doc in docs['forecast']:
                        temp_f = doc.get('temperature')
                        condition = doc.get('shortForecast', '')
                        wind = f"{doc.get('windSpeed', '')} {doc.get('windDirection', '')}".strip() if doc.get('windSpeed') else ''
                        precip = f"{doc.get('precipitation', 0)}%" if doc.get('precipitation') else ''
                        
                        city_html += f'''
                            <div class="doc-item">
                                <div class="doc-headline">{doc['headline']}</div>
                                <div class="doc-time">{doc['issued_at']}</div>
                                {f'<div class="forecast-temp"><span class="temp-value" data-fahrenheit="{temp_f}">{temp_f}</span><span class="temp-unit">°F</span></div>' if temp_f is not None else ''}
                                {f'<div class="forecast-condition">{condition}</div>' if condition else ''}
                                <div class="forecast-details">
                                    {f'<span>💨 {wind}</span>' if wind else ''}
                                    {f'<span>💧 {precip}</span>' if precip else ''}
                                </div>
                            </div>
                        '''
                else:
                    city_html += '<div class="empty-column">No forecasts</div>'
                
                city_html += f'''
                        </div>
                        <div class="column hourly">
                            <div class="column-header">⏰ Hourly ({len(docs['hourly_forecast'])})</div>
                '''
                
                if docs['hourly_forecast']:
                    for doc in docs['hourly_forecast']:
                        temp_f = doc.get('temperature')
                        condition = doc.get('shortForecast', '')
                        wind = f"{doc.get('windSpeed', '')} {doc.get('windDirection', '')}".strip() if doc.get('windSpeed') else ''
                        precip = f"{doc.get('precipitation', 0)}%" if doc.get('precipitation') else ''
                        
                        city_html += f'''
                            <div class="doc-item">
                                <div class="doc-time" style="font-weight: 600; font-size: 12px; color: #4CAF50;">{doc['issued_at']}</div>
                                {f'<div class="forecast-temp"><span class="temp-value" data-fahrenheit="{temp_f}">{temp_f}</span><span class="temp-unit">°F</span></div>' if temp_f is not None else ''}
                                {f'<div class="forecast-condition">{condition}</div>' if condition else ''}
                                <div class="forecast-details">
                                    {f'<span>💨 {wind}</span>' if wind else ''}
                                    {f'<span>💧 {precip}</span>' if precip else ''}
                                </div>
                            </div>
                        '''
                else:
                    city_html += '<div class="empty-column">No hourly forecasts</div>'
                
                city_html += '''
                        </div>
                    </div>
                </div>
                '''
        else:
            city_html += '<p>No documents found for this city.</p>'
        
        city_html += '''
            </div>
            <script>
                // Temperature conversion logic
                const tempToggle = document.getElementById('tempUnit');
                const fahrenheitLabel = document.getElementById('fahrenheitLabel');
                const celsiusLabel = document.getElementById('celsiusLabel');
                
                // Load saved preference from localStorage
                const savedUnit = localStorage.getItem('tempUnit') || 'F';
                if (savedUnit === 'C') {
                    tempToggle.checked = true;
                    convertTemperatures(true);
                    fahrenheitLabel.style.opacity = '0.5';
                    celsiusLabel.style.opacity = '1';
                }
                
                // Toggle event listener
                tempToggle.addEventListener('change', function() {
                    const toCelsius = this.checked;
                    convertTemperatures(toCelsius);
                    
                    // Update label opacity
                    fahrenheitLabel.style.opacity = toCelsius ? '0.5' : '1';
                    celsiusLabel.style.opacity = toCelsius ? '1' : '0.5';
                    
                    // Save preference
                    localStorage.setItem('tempUnit', toCelsius ? 'C' : 'F');
                });
                
                function convertTemperatures(toCelsius) {
                    const tempElements = document.querySelectorAll('.temp-value');
                    const unitElements = document.querySelectorAll('.temp-unit');
                    
                    tempElements.forEach(element => {
                        const fahrenheit = parseFloat(element.getAttribute('data-fahrenheit'));
                        
                        if (!isNaN(fahrenheit)) {
                            if (toCelsius) {
                                // Convert F to C: (F - 32) * 5/9
                                const celsius = Math.round((fahrenheit - 32) * 5 / 9);
                                element.textContent = celsius;
                            } else {
                                // Show original Fahrenheit
                                element.textContent = Math.round(fahrenheit);
                            }
                        }
                    });
                    
                    // Update unit symbols
                    unitElements.forEach(element => {
                        element.textContent = toCelsius ? '°C' : '°F';
                    });
                }
            </script>
        </body>
        </html>
        '''
        
        return city_html
        
    except Exception as e:
        return f"<h1>Error loading city data</h1><p>{str(e)}</p>", 500

@app.route('/weather/sync', methods=['POST'])
def weather_sync():
    """Trigger weather data sync from National Weather Service API
    
    Accepts JSON body:
        {
            "locations": ["Chicago, IL", "Austin, TX"],
            "limit": 50  # Max documents PER CITY (not total)
        }
    """
    try:
        # Get request body
        request_data = request.get_json() or {}
        
        # Extract locations and limit from request, with defaults from config
        locations = request_data.get('locations', config['weather']['locations'])
        limit = request_data.get('limit', config['weather']['limit'])
        
        # Validate locations
        if not locations or not isinstance(locations, list):
            return jsonify({
                'success': False,
                'error': 'locations must be a non-empty list'
            }), 400
        
        sync_config = {
            "locations": locations,
            "limit": limit,
            "include_alerts": config['weather'].get('include_alerts', True),
            "include_forecasts": config['weather'].get('include_forecasts', True),
            "include_hourly": config['weather'].get('include_hourly', True),
            "embeddings": config.get('embeddings', {}),  # Pass embedding configuration
        }
        
        result = sync_weather_data(sync_config)
        
        # Query per-city breakdown
        city_breakdown = []
        try:
            city_stats_query = f'''
                SELECT 
                    location,
                    COUNT(*) as doc_count,
                    COUNT(DISTINCT source_type) as doc_types
                FROM "{config['database']['schema']}".{config['database']['table']}
                GROUP BY location
                ORDER BY doc_count DESC
            '''
            city_stats = lakebase.run_query(city_stats_query)
            city_breakdown = [
                {
                    'location': row['location'],
                    'count': row['doc_count'],
                    'types': row['doc_types']
                }
                for row in city_stats
            ]
        except Exception as e:
            # If city breakdown query fails, continue without it
            pass
        
        return jsonify({
            'success': True,
            'documents_fetched': result['documents_fetched'],
            'documents_upserted': result['documents_upserted'],
            'chunks_embedded': result.get('chunks_embedded', 0),
            'stats': result['stats'],
            'city_breakdown': city_breakdown,
            'errors': result['errors'][:5] if result['errors'] else []
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/weather/search', methods=['POST'])
def weather_search():
    """Semantic search over weather embeddings using vector similarity
    
    Accepts JSON body:
        {
            "query": "risk of flooding near rivers",
            "top_k": 5
        }
    
    Returns:
        JSON array of top_k matches with location, headline, chunk_text, and similarity score
    """
    try:
        # Get request body
        request_data = request.get_json() or {}
        
        # Extract and validate query
        query = request_data.get('query', '').strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query string is required'
            }), 400
        
        # Extract and validate top_k (clamp to 1-20)
        top_k = request_data.get('top_k', 5)
        try:
            top_k = int(top_k)
            top_k = max(1, min(20, top_k))  # Clamp to [1, 20]
        except (ValueError, TypeError):
            top_k = 5
        
        # Load embedding model (cached singleton)
        model = get_embedding_model()
        
        # Embed the query
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()
        
        # Build the vector as a PostgreSQL array literal
        vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Run cosine similarity search
        # pgvector's <=> operator computes cosine distance (1 - cosine_similarity)
        # We'll convert it back to similarity for display
        search_query = f'''
            SELECT 
                e.embedding_id,
                e.chunk_text,
                e.chunk_index,
                d.location,
                d.headline,
                d.source_type,
                d.issued_at,
                (1 - (e.embedding <=> '{vector_str}'::vector)) as similarity
            FROM "{config['database']['schema']}".weather_embeddings e
            JOIN "{config['database']['schema']}".{config['database']['table']} d
                ON e.document_id = d.id
            ORDER BY e.embedding <=> '{vector_str}'::vector
            LIMIT {top_k}
        '''
        
        results = lakebase.run_query(search_query)
        
        # Handle empty embeddings table
        if not results:
            return jsonify({
                'success': True,
                'query': query,
                'top_k': top_k,
                'results': [],
                'message': 'No embeddings found. Please sync weather data first.'
            })
        
        # Format results
        formatted_results = [
            {
                'location': row['location'],
                'headline': row['headline'] or 'No headline',
                'chunk_text': row['chunk_text'],
                'source_type': row['source_type'],
                'issued_at': row['issued_at'].isoformat() if row['issued_at'] else None,
                'similarity': round(float(row['similarity']), 4)
            }
            for row in results
        ]
        
        return jsonify({
            'success': True,
            'query': query,
            'top_k': top_k,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': config.get('app', {}).get('version', '1.0.0')
    })

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        stats_query = f'''
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT location) as total_cities,
                COUNT(DISTINCT source_type) as total_types,
                MAX(synced_at) as last_sync,
                MIN(issued_at) as earliest_document,
                MAX(issued_at) as latest_document
            FROM "{config['database']['schema']}".{config['database']['table']}
        '''
        stats = lakebase.run_query(stats_query)[0]
        
        return jsonify({
            'success': True,
            'stats': {
                'total_documents': stats['total_documents'],
                'total_cities': stats['total_cities'],
                'total_types': stats['total_types'],
                'last_sync': stats['last_sync'].isoformat() if stats['last_sync'] else None,
                'earliest_document': stats['earliest_document'].isoformat() if stats['earliest_document'] else None,
                'latest_document': stats['latest_document'].isoformat() if stats['latest_document'] else None,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*80)
    print("🌤️  Weather Data Dashboard")
    print("="*80)
    print(f"\n🌐 Starting server on {config['app']['host']}:{config['app']['port']}")
    print(f"\n📊 Dashboard: http://{config['app']['host']}:{config['app']['port']}")
    print(f"📈 API Stats: http://{config['app']['host']}:{config['app']['port']}/api/stats")
    print(f"❤️  Health Check: http://{config['app']['host']}:{config['app']['port']}/health")
    print(f"\n🔄 Tracking {len(config['weather']['locations'])} cities")
    print(f"💾 Database: {config['database']['schema']}.{config['database']['table']}")
    print(f"\nPress Ctrl+C to stop the server")
    print("="*80)
    print()
    
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug'],
        use_reloader=False
    )