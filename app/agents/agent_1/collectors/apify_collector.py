"""
Apify Collector Implementation
Handles data collection using Apify actors
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False


class ApifyCollector:
    """Apify collector implementation following detailed specification"""
    
    def __init__(self, client):
        self.client = client
    
    async def collect_competitor_data(self, competitor_url: str = None) -> List[Dict[str, Any]]:
        """Collect competitor data using Apify actors"""
        if not self.client:
            return []
        
        try:
            run_input = {
                'startUrls': [{'url': competitor_url or 'https://example-competitor.com'}],
                'maxCrawlingDepth': 2,
                'maxResultsPerPage': 100,
                'proxyConfiguration': {'useApifyProxy': True}
            }
            
            run = self.client.actor('apify/website-scraper').call(run_input=run_input)
            
            items = []
            for item in self.client.dataset(run['defaultDatasetId']).iterate_items():
                items.append({
                    'source_type': 'competitor',
                    'entity': self._extract_domain(item.get('url', '')),
                    'signal_type': 'competitor_intelligence',
                    'content': item.get('text', ''),
                    'metadata': {
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                        'collected_at': datetime.now().isoformat(),
                        'method': 'apify'
                    }
                })
            
            return items
            
        except Exception as e:
            print(f"Apify competitor scraping failed: {e}")
            return []
    
    async def collect_review_data(self, app_url: str = None) -> List[Dict[str, Any]]:
        """Collect review data"""
        if not self.client:
            return []
        
        try:
            run_input = {
                'appIds': [app_url or 'com.example.app'],
                'country': 'us',
                'device': 'iphone',
                'lang': 'en'
            }
            
            run = self.client.actor('junglee/app-store-scraper').call(run_input=run_input)
            
            items = []
            for item in self.client.dataset(run['defaultDatasetId']).iterate_items():
                items.append({
                    'source_type': 'user',
                    'entity': item.get('appId', ''),
                    'signal_type': 'review',
                    'content': item.get('reviewText', ''),
                    'metadata': {
                        'rating': item.get('rating', 0),
                        'date': item.get('date', ''),
                        'version': item.get('version', ''),
                        'collected_at': datetime.now().isoformat(),
                        'method': 'apify'
                    }
                })
            
            return items
            
        except Exception as e:
            print(f"Apify review collection failed: {e}")
            return []
    
    async def collect_news_data(self, search_query: str = None) -> List[Dict[str, Any]]:
        """Collect news data"""
        if not self.client:
            return []
        
        try:
            run_input = {
                'queries': [search_query or 'startup news technology'],
                'maxResultsPerPage': 50,
                'language': 'en'
            }
            
            run = self.client.actor('apify/google-news-scraper').call(run_input=run_input)
            
            items = []
            for item in self.client.dataset(run['defaultDatasetId']).iterate_items():
                items.append({
                    'source_type': 'news',
                    'entity': item.get('source', {}).get('name', ''),
                    'signal_type': 'trend',
                    'content': item.get('content', ''),
                    'metadata': {
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                        'published_at': item.get('publishedAt', ''),
                        'collected_at': datetime.now().isoformat(),
                        'method': 'apify'
                    }
                })
            
            return items
            
        except Exception as e:
            print(f"Apify news collection failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return url
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Apify collector"""
        return {
            'client_available': self.client is not None,
            'token_valid': bool(self.client and self.client.token if hasattr(self.client, 'token') else False)
        }
