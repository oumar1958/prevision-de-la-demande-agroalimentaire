"""
Real Web scraper for Carrefour.fr agroalimentary product data
Scrapes real prices, promotions, availability from Carrefour website
"""

import math
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse, quote
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarrefourScraper:
    """
    Real scraper for Carrefour.fr product data
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://www.carrefour.fr"
        self.search_url = "https://www.carrefour.fr/r?query="
        
        # Realistic headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        self.delay = 2.0  # Increased delay for real scraping
        self.timeout = 30
        self.max_retries = 3
        
        # Product categories for Carrefour
        self.product_categories = {
            'fruits_legumes': [
                'pomme', 'banane', 'carotte', 'tomate', 'salade', 
                'pomme de terre', 'oignon', 'poivron', 'courgette', 'aubergine'
            ],
            'produits_laitiers': [
                'lait', 'yaourt', 'fromage', 'beurre', 'crème fraîche',
                'emmental', 'camembert', 'yoplait', 'danone'
            ],
            'viandes': [
                'poulet', 'bœuf', 'porc', 'agneau', 'steak',
                'côte de porc', 'blanc de poulet', 'haché'
            ],
            'cereales': [
                'pain', 'pâtes', 'riz', 'farine', 'céréales',
                'baguette', 'spaghetti', 'penne', 'riz basmati'
            ]
        }
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and rate limiting"""
        for attempt in range(self.max_retries):
            try:
                # Add random delay to avoid detection
                time.sleep(self.delay + random.uniform(0.5, 1.5))
                
                logger.info(f"Attempting to fetch: {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Check if we got a valid page
                if "carrefour" in response.text.lower():
                    return response
                else:
                    logger.warning(f"Response doesn't contain Carrefour content")
                    
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _extract_product_data_from_html(self, html_content: str, search_term: str, category: str) -> List[Dict]:
        """Extract product information from Carrefour HTML"""
        products = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Carrefour product cards (these selectors may need updating)
            product_cards = soup.find_all('div', {'data-test': 'product-card'}) or \
                           soup.find_all('article', class_='product-card') or \
                           soup.find_all('div', class_='product-item')
            
            logger.info(f"Found {len(product_cards)} product cards for {search_term}")
            
            for i, card in enumerate(product_cards):
                try:
                    # Product name
                    name_elem = card.find('h2') or card.find('h3') or card.find('a', class_='product-title')
                    if not name_elem:
                        continue
                    
                    name = name_elem.get_text(strip=True)
                    
                    # Price extraction
                    price_elem = card.find('span', {'data-test': 'product-price'}) or \
                                card.find('span', class_='price') or \
                                card.find('div', class_='product-price')
                    
                    if not price_elem:
                        continue
                    
                    price_text = price_elem.get_text(strip=True)
                    price = self._parse_price(price_text)
                    
                    # Original price (for promotions)
                    original_price_elem = card.find('span', class_='original-price') or \
                                        card.find('s', class_='price-strikethrough')
                    
                    original_price = None
                    if original_price_elem:
                        orig_price_text = original_price_elem.get_text(strip=True)
                        original_price = self._parse_price(orig_price_text)
                    
                    # Promotion detection
                    promo_elem = card.find('span', class_='promotion') or \
                                card.find('div', class_='promo-badge') or \
                                card.find('span', class_='discount')
                    
                    is_promo = promo_elem is not None
                    promo_percentage = self._calculate_promo_percentage(price, original_price) if is_promo else 0
                    
                    # Availability
                    stock_elem = card.find('span', class_='stock') or \
                               card.find('div', class_='availability') or \
                               card.find('button', class_='add-to-cart')
                    
                    stock_text = stock_elem.get_text(strip=True).lower() if stock_elem else ""
                    is_available = not any(word in stock_text for word in ['rupture', 'indisponible', 'bientôt'])
                    
                    # Product image URL (optional)
                    img_elem = card.find('img')
                    image_url = img_elem.get('src') if img_elem else None
                    
                    product_data = {
                        'product_name': name,
                        'category': category,
                        'current_price': price,
                        'original_price': original_price,
                        'is_promotion': is_promo,
                        'promotion_percentage': promo_percentage,
                        'is_available': is_available,
                        'scraped_at': datetime.now(),
                        'retailer': 'Carrefour',
                        'search_term': search_term,
                        'image_url': image_url,
                        'product_url': self.base_url
                    }
                    
                    products.append(product_data)
                    
                    # Limit to first 10 products per search to avoid overwhelming
                    if len(products) >= 10:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error extracting product data: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
        
        return products
    
    def _parse_price(self, price_text: str) -> float:
        """Parse price string to float"""
        try:
            # Remove currency symbols and whitespace
            price_clean = price_text.replace('€', '').replace('EUR', '').strip()
            price_clean = price_clean.replace(',', '.')
            
            # Extract numeric value using regex
            import re
            match = re.search(r'[\d.]+', price_clean)
            if match:
                return float(match.group())
        except Exception:
            pass
        return 0.0
    
    def _calculate_promo_percentage(self, current_price: float, original_price: float) -> float:
        """Calculate promotion percentage"""
        if original_price and original_price > 0 and current_price < original_price:
            return round((1 - current_price / original_price) * 100, 2)
        return 0.0
    
    def scrape_product_category(self, category: str, products: List[str]) -> List[Dict]:
        """Scrape data for a specific product category from Carrefour"""
        all_products = []
        
        logger.info(f"Scraping Carrefour category: {category}")
        
        for product in products:
            try:
                # Construct search URL
                encoded_product = quote(product)
                search_url = f"{self.search_url}{encoded_product}"
                
                logger.info(f"Scraping {product} from Carrefour")
                
                response = self._make_request(search_url)
                if not response:
                    logger.warning(f"No response for {product}")
                    continue
                
                # Extract product data
                products_data = self._extract_product_data_from_html(
                    response.text, product, category
                )
                
                all_products.extend(products_data)
                logger.info(f"Found {len(products_data)} products for {product}")
                
            except Exception as e:
                logger.error(f"Error scraping {product} from Carrefour: {e}")
                continue
        
        return all_products
    
    def scrape_all_categories(self) -> pd.DataFrame:
        """Scrape all product categories from Carrefour"""
        all_data = []
        
        for category, products in self.product_categories.items():
            category_data = self.scrape_product_category(category, products)
            all_data.extend(category_data)
            
            # Longer delay between categories
            time.sleep(3)
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Successfully scraped {len(df)} products from Carrefour")
            return df
        else:
            logger.warning("No data scraped from Carrefour")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save scraped data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"carrefour_products_{timestamp}.csv"
        
        filepath = f"data/raw/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Carrefour data saved to {filepath}")
        return filepath


def main():
    """Test the Carrefour scraper"""
    scraper = CarrefourScraper()
    
    # Test with one category first
    logger.info("Starting Carrefour scraping test...")
    
    # Scrape fruits and vegetables as test
    test_category = 'fruits_legumes'
    test_products = ['pomme', 'banane', 'tomate']
    
    results = scraper.scrape_product_category(test_category, test_products)
    
    if results:
        df = pd.DataFrame(results)
        print(f"Successfully scraped {len(df)} products:")
        print(df.head())
        
        # Save results
        scraper.save_data(df)
    else:
        print("No products scraped")


if __name__ == "__main__":
    main()
