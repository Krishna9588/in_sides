"""
Data Enricher Implementation
Enriches data with additional metadata
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime


class DataEnricher:
    """Data enricher implementation following detailed specification"""
    
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich data with additional metadata"""
        for item in data:
            if 'metadata' not in item:
                item['metadata'] = {}
            
            item['metadata']['processed_at'] = datetime.now().isoformat()
            item['metadata']['word_count'] = len(item.get('content', '').split())
            item['metadata']['char_count'] = len(item.get('content', ''))
        
        return data
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for data enricher"""
        return {'status': 'working'}
