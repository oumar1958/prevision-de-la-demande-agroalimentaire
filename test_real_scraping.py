"""
Test script for real scraping functionality
Tests Carrefour.fr and real weather data collection
"""

import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_carrefour_scraper import CarrefourScraper
from real_weather_scraper import RealWeatherScraper
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_carrefour_scraping():
    """Test Carrefour scraping with real data"""
    print("=" * 60)
    print("🛒 TEST CARREFOUR SCRAPING")
    print("=" * 60)
    
    scraper = CarrefourScraper()
    
    # Test with a few products first
    test_products = ['pomme', 'lait']
    
    for product in test_products:
        print(f"\n🔍 Testing product: {product}")
        
        try:
            # Construct search URL
            from urllib.parse import quote
            encoded_product = quote(product)
            search_url = f"{scraper.search_url}{encoded_product}"
            
            print(f"URL: {search_url}")
            
            # Make request
            response = scraper._make_request(search_url)
            
            if response:
                print(f"✅ Response received: {len(response.text)} characters")
                
                # Check if we got Carrefour content
                if "carrefour" in response.text.lower():
                    print("✅ Carrefour content detected")
                    
                    # Extract products
                    products = scraper._extract_product_data_from_html(
                        response.text, product, 'test'
                    )
                    
                    print(f"📦 Found {len(products)} products")
                    
                    if products:
                        df = pd.DataFrame(products)
                        print("\nSample products:")
                        print(df[['product_name', 'current_price', 'is_promotion', 'is_available']].head())
                        
                        # Save test data
                        scraper.save_data(df, f"test_carrefour_{product}.csv")
                    else:
                        print("⚠️ No products extracted - selectors may need updating")
                else:
                    print("❌ No Carrefour content detected")
            else:
                print("❌ No response received")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def test_weather_scraping():
    """Test real weather data scraping"""
    print("\n" + "=" * 60)
    print("🌤️ TEST REAL WEATHER SCRAPING")
    print("=" * 60)
    
    scraper = RealWeatherScraper()
    
    # Test with recent data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    print(f"📅 Period: {start_date} to {end_date}")
    
    try:
        # Test historical weather for Paris
        print("\n🌍 Testing Paris weather...")
        weather_df = scraper.get_historical_weather(start_date, end_date, 'paris')
        
        if not weather_df.empty:
            print(f"✅ Retrieved {len(weather_df)} weather records")
            print("\nSample weather data:")
            print(weather_df[['date', 'temperature_2m', 'precipitation', 'humidity']].head())
            
            # Save test data
            scraper.save_weather_data(weather_df, "test_weather_paris.csv")
            
            # Test current weather
            print("\n🌡️ Testing current weather...")
            current = scraper.get_current_weather('paris')
            print(f"Current weather: {current}")
            
        else:
            print("❌ No weather data retrieved")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_integration():
    """Test integration of both scrapers"""
    print("\n" + "=" * 60)
    print("🔗 INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Test both scrapers
        print("🔄 Running both scrapers...")
        
        # Quick test with minimal data
        carrefour_scraper = CarrefourScraper()
        weather_scraper = RealWeatherScraper()
        
        # Test one product
        carrefour_results = carrefour_scraper.scrape_product_category(
            'fruits_legumes', ['pomme']
        )
        
        # Test one day of weather
        today = datetime.now().strftime('%Y-%m-%d')
        weather_results = weather_scraper.get_historical_weather(today, today, 'paris')
        
        print(f"📊 Results:")
        print(f"  Carrefour products: {len(carrefour_results)}")
        print(f"  Weather records: {len(weather_results)}")
        
        if carrefour_results and not weather_results.empty:
            print("✅ Integration test successful!")
            
            # Create combined dataset
            products_df = pd.DataFrame(carrefour_results)
            
            # Add weather data to products (simplified)
            if not weather_results.empty:
                avg_temp = weather_results['temperature_2m'].mean()
                products_df['temperature'] = avg_temp
                products_df['weather_date'] = weather_results['date'].iloc[0]
            
            print("\n📋 Combined dataset sample:")
            print(products_df[['product_name', 'current_price', 'temperature']].head())
            
        else:
            print("❌ Integration test failed")
            
    except Exception as e:
        print(f"❌ Integration error: {e}")


def main():
    """Main test function"""
    print("🚀 STARTING REAL SCRAPING TESTS")
    print(f"⏰ Time: {datetime.now()}")
    
    # Run tests
    test_carrefour_scraping()
    test_weather_scraping()
    test_integration()
    
    print("\n" + "=" * 60)
    print("✅ TESTS COMPLETED")
    print("=" * 60)
    
    print("\n📝 NOTES:")
    print("- If Carrefour scraping fails, selectors may need updating")
    print("- Weather scraping uses free Open-Meteo API")
    print("- Both scrapers include rate limiting for ethical use")
    print("- Check data/raw/ for saved test results")


if __name__ == "__main__":
    main()
