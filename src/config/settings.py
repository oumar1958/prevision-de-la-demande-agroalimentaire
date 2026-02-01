"""
Configuration settings for the agro demand forecasting project
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Database configuration
DATABASE_URL = f"sqlite:///{DATA_DIR}/agro_forecasting.db"

# Scraping configuration
SCRAPING_CONFIG = {
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    },
    "delay_between_requests": 1.0,  # seconds
    "timeout": 30,
    "max_retries": 3
}

# Target products for scraping
PRODUCT_CATEGORIES = {
    "fruits_legumes": [
        "pommes", "bananes", "carottes", "tomates", "salades", 
        "pommes_de_terre", "oignons", "poivrons"
    ],
    "produits_laitiers": [
        "lait", "yaourts", "fromage", "beurre", "crème"
    ],
    "viandes": [
        "poulet", "bœuf", "porc", "agneau"
    ],
    "cereales": [
        "pain", "pâtes", "riz", "farine"
    ]
}

# Weather API configuration
WEATHER_CONFIG = {
    "base_url": "https://api.open-meteo.com/v1/forecast",
    "historical_url": "https://archive-api.open-meteo.com/v1/archive",
    "parameters": ["temperature_2m", "precipitation", "humidity", "wind_speed"],
    "location": {
        "latitude": 48.8566,  # Paris
        "longitude": 2.3522
    }
}

# Model configuration
MODEL_CONFIG = {
    "prophet": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "lstm": {
        "sequence_length": 30,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2
    }
}

# Business simulation parameters
BUSINESS_CONFIG = {
    "storage_cost_per_unit": 0.05,  # € per unit per day
    "shortage_cost_per_unit": 2.0,  # € per unit
    "waste_cost_per_unit": 1.5,     # € per unit
    "production_capacity_multiplier": 1.2
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": f"{DATA_DIR}/logs/agro_forecasting.log"
}
