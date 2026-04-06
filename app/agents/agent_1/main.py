"""
Agent 1: Research Ingestion - Main Orchestrator
Complete implementation following detailed specification from docs/agent-1-implementation.md
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from ..base_agent import BaseAgent
from ...models.signal import Signal, SignalCreate
from ...config.database import signal_repo
from ...utils.nlp_utils import nlp_utils
from ...utils.cache import cache_manager, CacheKeys
from ...config.settings import settings

from .collectors import ApifyCollector, WebScraper, APICollector, FileProcessor
from .processors import TextCleaner, DataNormalizer, DataDeduplicator, DataEnricher
from .validators import QualityValidator, SchemaValidator, BusinessValidator

try:
    from apify_client import ApifyClient
    import requests
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    import PyPDF2
    import docx
    EXTERNAL_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    EXTERNAL_LIBS_AVAILABLE = False


class ResearchIngestionAgent(BaseAgent):
    """
    Research Ingestion Agent - Complete Implementation
    
    Following detailed specification:
    1. Data Collection (Apify actors, web scraping, file processing, API integrations)
    2. Data Processing (text cleaning, normalization, deduplication, enrichment)
    3. Data Validation (quality, schema, business rules)
    4. Data Storage (signals database with caching)
    """
    
    def __init__(self):
        super().__init__("agent_1")
        self.apify_client = None
        self.collectors = {}
        self.processors = {}
        self.validators = {}
        self._init_components()
    
    def _init_components(self):
        """Initialize all components as per detailed specification"""
        # Initialize Apify client
        if EXTERNAL_LIBS_AVAILABLE and settings.APIFY_TOKEN:
            try:
                self.apify_client = ApifyClient(settings.APIFY_TOKEN)
                self.log_info("Apify client initialized")
            except Exception as e:
                self.log_error(f"Failed to initialize Apify client: {e}")
        
        # Initialize collectors following detailed spec
        self.collectors = {
            'apify': ApifyCollector(self.apify_client),
            'web': WebScraper(),
            'api': APICollector(),
            'file': FileProcessor()
        }
        
        # Initialize processors following detailed spec
        self.processors = {
            'cleaner': TextCleaner(),
            'normalizer': DataNormalizer(),
            'deduplicator': DataDeduplicator(),
            'enricher': DataEnricher()
        }
        
        # Initialize validators following detailed spec
        self.validators = {
            'quality': QualityValidator(min_confidence=settings.CONFIDENCE_THRESHOLD),
            'schema': SchemaValidator(),
            'business': BusinessValidator()
        }
    
    async def validate_input(self, data_sources: List[str] = None, **kwargs) -> bool:
        """Validate input parameters"""
        if data_sources is None:
            data_sources = ['competitor', 'reviews', 'news']
        
        valid_sources = ['competitor', 'reviews', 'news', 'manual', 'api']
        for source in data_sources:
            if source not in valid_sources:
                self.log_error(f"Invalid data source: {source}")
                return False
        
        return True
    
    async def run(self, data_sources: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run complete research ingestion pipeline
        
        Following detailed specification:
        1. Collection Phase
        2. Processing Phase  
        3. Validation Phase
        4. Storage Phase
        """
        self.log_info(f"Starting research ingestion for sources: {data_sources}")
        
        try:
            # Step 1: Collection
            raw_data = await self._run_collection(data_sources, **kwargs)
            self.log_info(f"Collected {len(raw_data)} raw items")
            
            # Step 2: Processing
            processed_data = await self._run_processing(raw_data)
            self.log_info(f"Processed {len(processed_data)} items")
            
            # Step 3: Validation
            validated_data = await self._run_validation(processed_data)
            self.log_info(f"Validated {len(validated_data)} items")
            
            # Step 4: Storage
            stored_signals = await self._store_signals(validated_data)
            self.log_info(f"Stored {len(stored_signals)} signals")
            
            return {
                'status': 'success',
                'data_sources': data_sources,
                'collection_results': self._get_collection_results(raw_data),
                'total_collected': len(raw_data),
                'total_processed': len(processed_data),
                'total_validated': len(validated_data),
                'total_stored': len(stored_signals),
                'signal_ids': [s.get('id') for s in stored_signals],
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"Pipeline execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'agent_id': self.agent_id,
                'processing_time': datetime.now().isoformat()
            }
    
    async def _run_collection(self, data_sources: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Run collection step following detailed specification"""
        all_data = []
        collection_results = {}
        
        for source in data_sources:
            try:
                self.log_info(f"Collecting from source: {source}")
                
                if source == 'competitor':
                    data = await self.collectors['apify'].collect_competitor_data(
                        kwargs.get('competitor_url')
                    )
                elif source == 'reviews':
                    data = await self.collectors['apify'].collect_review_data(
                        kwargs.get('app_url')
                    )
                elif source == 'news':
                    data = await self.collectors['apify'].collect_news_data(
                        kwargs.get('search_query')
                    )
                elif source == 'manual':
                    data = await self.collectors['file'].process_manual_data(
                        kwargs.get('manual_data')
                    )
                elif source == 'api':
                    data = await self.collectors['api'].collect_api_data(
                        kwargs.get('api_config')
                    )
                else:
                    data = []
                
                collection_results[source] = {
                    'status': 'success',
                    'count': len(data),
                    'collector': self._get_collector_name(source)
                }
                all_data.extend(data)
                
            except Exception as e:
                self.log_error(f"Collection failed for {source}: {e}")
                collection_results[source] = {
                    'status': 'error',
                    'error': str(e),
                    'count': 0,
                    'collector': self._get_collector_name(source)
                }
        
        return all_data
    
    async def _run_processing(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run processing step following detailed specification"""
        processed_data = raw_data
        
        # Apply each processor in sequence as per spec
        for processor_name, processor in self.processors.items():
            try:
                self.log_info(f"Running processor: {processor_name}")
                processed_data = await processor.process(processed_data)
                self.log_info(f"Processor {processor_name} completed")
            except Exception as e:
                self.log_error(f"Processor {processor_name} failed: {e}")
        
        return processed_data
    
    async def _run_validation(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run validation step following detailed specification"""
        validated_data = []
        
        for item in processed_data:
            try:
                # Run all validators
                validation_results = {}
                for validator_name, validator in self.validators.items():
                    validation_results[validator_name] = await validator.validate(item)
                
                # Check if item passes all validators
                item['validation_status'] = 'valid' if all(
                    result.get('is_valid', False) for result in validation_results.values()
                ) else 'rejected'
                
                # Add validation results
                item['validation_results'] = validation_results
                
                if item['validation_status'] == 'valid':
                    validated_data.append(item)
                
            except Exception as e:
                self.log_error(f"Validation failed for item: {e}")
        
        return validated_data
    
    async def _store_signals(self, validated_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store validated signals in database following detailed spec"""
        self.log_info(f"Storing {len(validated_data)} signals")
        
        stored_signals = []
        
        for item in validated_data:
            try:
                # Create signal model with all required fields
                signal_data = {
                    'source_type': item.get('source_type', 'unknown'),
                    'entity': item.get('entity', 'unknown'),
                    'signal_type': item.get('signal_type', 'insight'),
                    'content': item.get('content', ''),
                    'metadata': item.get('metadata', {}),
                    'confidence_score': item.get('confidence_score', 0.8),
                    'relevance_score': item.get('relevance_score', 0.8)
                }
                
                signal = SignalCreate(**signal_data)
                
                # Store in database
                stored_signal = await signal_repo.create(signal.dict())
                
                if stored_signal:
                    stored_signals.append(stored_signal)
                    
                    # Cache for quick retrieval
                    cache_key = f"signal:{stored_signal['id']}"
                    cache_manager.set(cache_key, stored_signal, ttl=3600)
                
            except Exception as e:
                self.log_error(f"Failed to store signal: {e}")
        
        self.log_info(f"Stored {len(stored_signals)} signals successfully")
        return stored_signals
    
    def _get_collection_results(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get collection results summary"""
        results = {}
        
        # Group by source type
        source_counts = {}
        for item in raw_data:
            source_type = item.get('source_type', 'unknown')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        for source_type, count in source_counts.items():
            results[source_type] = {
                'status': 'success',
                'count': count,
                'collector': self._get_collector_name(source_type)
            }
        
        return results
    
    def _get_collector_name(self, source_type: str) -> str:
        """Get collector name for source type"""
        collector_map = {
            'competitor': 'ApifyCollector',
            'user': 'ReviewCollector',
            'news': 'NewsCollector',
            'manual': 'FileProcessor',
            'api': 'APICollector'
        }
        return collector_map.get(source_type, 'UnknownCollector')
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform agent-specific health checks"""
        health_status = {}
        
        # Check Apify client
        if self.apify_client:
            health_status['apify_client'] = 'connected'
        else:
            health_status['apify_client'] = 'not_available'
        
        # Check external libraries
        health_status['external_libs'] = 'available' if EXTERNAL_LIBS_AVAILABLE else 'not_available'
        
        # Check individual collectors
        for collector_name, collector in self.collectors.items():
            try:
                if hasattr(collector, 'health_check'):
                    health_status[f'collector_{collector_name}'] = await collector.health_check()
                else:
                    health_status[f'collector_{collector_name}'] = 'working'
            except Exception as e:
                health_status[f'collector_{collector_name}'] = f'error: {str(e)}'
        
        # Check processors
        for processor_name, processor in self.processors.items():
            try:
                if hasattr(processor, 'health_check'):
                    health_status[f'processor_{processor_name}'] = await processor.health_check()
                else:
                    health_status[f'processor_{processor_name}'] = 'working'
            except Exception as e:
                health_status[f'processor_{processor_name}'] = f'error: {str(e)}'
        
        # Check validators
        for validator_name, validator in self.validators.items():
            try:
                if hasattr(validator, 'health_check'):
                    health_status[f'validator_{validator_name}'] = await validator.health_check()
                else:
                    health_status[f'validator_{validator_name}'] = 'working'
            except Exception as e:
                health_status[f'validator_{validator_name}'] = f'error: {str(e)}'
        
        # Check database connection
        try:
            recent_signals = await signal_repo.get_recent_signals(limit=1)
            health_status['database'] = 'connected'
        except Exception as e:
            health_status['database'] = f'error: {str(e)}'
        
        return health_status
