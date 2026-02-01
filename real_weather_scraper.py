"""
Real weather data scraper using Open-Meteo API
Gets actual historical and current weather data for France
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealWeatherScraper:
    """
    Real weather data scraper using Open-Meteo API
    """
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.current_url = "https://api.open-meteo.com/v1/forecast"
        
        # French cities coordinates (major agricultural regions)
        self.locations = {
            'paris': {'latitude': 48.8566, 'longitude': 2.3522, 'name': 'Paris Île-de-France'},
            'lyon': {'latitude': 45.7640, 'longitude': 4.8357, 'name': 'Lyon Rhône-Alpes'},
            'marseille': {'latitude': 43.2965, 'longitude': 5.3698, 'name': 'Marseille Provence'},
            'bordeaux': {'latitude': 44.8378, 'longitude': -0.5792, 'name': 'Bordeaux Aquitaine'},
            'toulouse': {'latitude': 43.6047, 'longitude': 1.4442, 'name': 'Toulouse Occitanie'},
            'nantes': {'latitude': 47.2184, 'longitude': -1.5536, 'name': 'Nantes Pays de Loire'},
            'lille': {'latitude': 50.6292, 'longitude': 3.0573, 'name': 'Lille Hauts-de-France'},
            'strasbourg': {'latitude': 48.5846, 'longitude': 7.7507, 'name': 'Strasbourg Grand Est'}
        }
        
        self.parameters = [
            "temperature_2m",
            "precipitation", 
            "humidity",
            "wind_speed",
            "pressure_msl",
            "cloud_cover"
        ]
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AgroDemandForecasting/1.0'
        })
    
    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            time.sleep(0.5)  # Rate limiting
            
            logger.info(f"Fetching weather data from: {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                logger.error(f"API Error: {data['error']}")
                return None
                
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
    
    def get_historical_weather(self, start_date: str, end_date: str, location: str = 'paris') -> pd.DataFrame:
        """
        Get historical weather data from Open-Meteo API
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            location: Location key from self.locations
            
        Returns:
            DataFrame with weather data
        """
        if location not in self.locations:
            logger.error(f"Unknown location: {location}")
            return pd.DataFrame()
        
        loc = self.locations[location]
        
        params = {
            "latitude": loc['latitude'],
            "longitude": loc['longitude'],
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(self.parameters),
            "timezone": "Europe/Paris"
        }
        
        logger.info(f"Fetching historical weather for {loc['name']} from {start_date} to {end_date}")
        
        data = self._make_request(self.base_url, params)
        if not data:
            return pd.DataFrame()
        
        # Extract daily data
        daily_data = data.get("daily", {})
        
        # Create DataFrame
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(daily_data.get("time", [])),
            'temperature_2m': daily_data.get("temperature_2m", []),
            'precipitation': daily_data.get("precipitation", []),
            'humidity': daily_data.get("humidity", []),
            'wind_speed': daily_data.get("wind_speed", []),
            'pressure_msl': daily_data.get("pressure_msl", []),
            'cloud_cover': daily_data.get("cloud_cover", [])
        })
        
        # Add location and derived features
        weather_df['location'] = loc['name']
        weather_df['latitude'] = loc['latitude']
        weather_df['longitude'] = loc['longitude']
        
        # Add derived features
        weather_df['year'] = weather_df['date'].dt.year
        weather_df['month'] = weather_df['date'].dt.month
        weather_df['day_of_year'] = weather_df['date'].dt.dayofyear
        weather_df['season'] = weather_df['month'].apply(self._get_season)
        
        # Weather conditions
        weather_df['is_rainy'] = weather_df['precipitation'] > 0.1
        weather_df['is_hot'] = weather_df['temperature_2m'] > 25
        weather_df['is_cold'] = weather_df['temperature_2m'] < 5
        weather_df['is_cloudy'] = weather_df['cloud_cover'] > 70
        
        logger.info(f"Successfully retrieved {len(weather_df)} days of weather data for {loc['name']}")
        return weather_df
    
    def get_current_weather(self, location: str = 'paris') -> Dict:
        """Get current weather conditions"""
        if location not in self.locations:
            logger.error(f"Unknown location: {location}")
            return {}
        
        loc = self.locations[location]
        
        params = {
            "latitude": loc['latitude'],
            "longitude": loc['longitude'],
            "current": ",".join(self.parameters),
            "timezone": "Europe/Paris"
        }
        
        logger.info(f"Fetching current weather for {loc['name']}")
        
        data = self._make_request(self.current_url, params)
        if not data:
            return {}
        
        current_data = data.get("current", {})
        
        return {
            'timestamp': datetime.now(),
            'location': loc['name'],
            'temperature_2m': current_data.get("temperature_2m"),
            'precipitation': current_data.get("precipitation"),
            'humidity': current_data.get("humidity"),
            'wind_speed': current_data.get("wind_speed"),
            'pressure_msl': current_data.get("pressure_msl"),
            'cloud_cover': current_data.get("cloud_cover")
        }
    
    def get_weather_forecast(self, days: int = 7, location: str = 'paris') -> pd.DataFrame:
        """
        Get weather forecast for the next N days
        
        Args:
            days: Number of days to forecast
            location: Location key
            
        Returns:
            DataFrame with forecast data
        """
        if location not in self.locations:
            logger.error(f"Unknown location: {location}")
            return pd.DataFrame()
        
        loc = self.locations[location]
        
        params = {
            "latitude": loc['latitude'],
            "longitude": loc['longitude'],
            "daily": ",".join(self.parameters),
            "timezone": "Europe/Paris",
            "forecast_days": days
        }
        
        logger.info(f"Fetching {days}-day weather forecast for {loc['name']}")
        
        data = self._make_request(self.current_url, params)
        if not data:
            return pd.DataFrame()
        
        # Extract daily forecast data
        daily_data = data.get("daily", {})
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'date': pd.to_datetime(daily_data.get("time", [])),
            'temperature_2m': daily_data.get("temperature_2m", []),
            'precipitation': daily_data.get("precipitation", []),
            'humidity': daily_data.get("humidity", []),
            'wind_speed': daily_data.get("wind_speed", []),
            'pressure_msl': daily_data.get("pressure_msl", []),
            'cloud_cover': daily_data.get("cloud_cover", [])
        })
        
        # Add location and derived features
        forecast_df['location'] = loc['name']
        forecast_df['is_forecast'] = True
        forecast_df['year'] = forecast_df['date'].dt.year
        forecast_df['month'] = forecast_df['date'].dt.month
        forecast_df['day_of_year'] = forecast_df['date'].dt.dayofyear
        forecast_df['season'] = forecast_df['month'].apply(self._get_season)
        
        logger.info(f"Successfully retrieved {len(forecast_df)} days of forecast data for {loc['name']}")
        return forecast_df
    
    def get_multiple_locations_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get weather data for multiple French locations"""
        all_weather_data = []
        
        for location_key in self.locations.keys():
            logger.info(f"Fetching weather for {location_key}")
            
            weather_df = self.get_historical_weather(start_date, end_date, location_key)
            if not weather_df.empty:
                all_weather_data.append(weather_df)
            
            # Rate limiting between locations
            time.sleep(1)
        
        if all_weather_data:
            combined_df = pd.concat(all_weather_data, ignore_index=True)
            logger.info(f"Successfully retrieved {len(combined_df)} weather records for all locations")
            return combined_df
        else:
            return pd.DataFrame()
    
    def _get_season(self, month: int) -> str:
        """Get season from month number"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def save_weather_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save weather data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_weather_data_{timestamp}.csv"
        
        filepath = f"data/raw/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Real weather data saved to {filepath}")
        return filepath


def main():
    """Test the real weather scraper"""
    scraper = RealWeatherScraper()
    
    # Test with recent data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info("Testing real weather scraper...")
    
    # Get weather for Paris
    weather_df = scraper.get_historical_weather(start_date, end_date, 'paris')
    
    if not weather_df.empty:
        print(f"Successfully retrieved {len(weather_df)} weather records:")
        print(weather_df.head())
        
        # Save results
        scraper.save_weather_data(weather_df)
        
        # Test current weather
        current_weather = scraper.get_current_weather('paris')
        print(f"\nCurrent weather in Paris: {current_weather}")
        
    else:
        print("No weather data retrieved")


if __name__ == "__main__":
    main()
