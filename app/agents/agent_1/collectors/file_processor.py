"""
File Processor Implementation
Handles processing of manually uploaded files
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import PyPDF2
    import docx
    FILE_PROCESSING_AVAILABLE = True
except ImportError:
    FILE_PROCESSING_AVAILABLE = False


class FileProcessor:
    """File processor implementation following detailed specification"""
    
    async def process_manual_data(self, manual_data: Any) -> List[Dict[str, Any]]:
        """Process manually uploaded data"""
        if not manual_data:
            return []
        
        items = []
        
        if isinstance(manual_data, list):
            for data in manual_data:
                items.append({
                    'source_type': 'manual',
                    'entity': data.get('entity', 'manual_upload'),
                    'signal_type': data.get('signal_type', 'insight'),
                    'content': data.get('content', ''),
                    'metadata': {
                        'file_type': data.get('file_type', 'unknown'),
                        'uploaded_at': datetime.now().isoformat(),
                        'method': 'manual'
                    }
                })
        elif isinstance(manual_data, dict):
            items.append({
                'source_type': 'manual',
                'entity': manual_data.get('entity', 'manual_upload'),
                'signal_type': manual_data.get('signal_type', 'insight'),
                'content': manual_data.get('content', ''),
                'metadata': {
                    'file_type': manual_data.get('file_type', 'unknown'),
                    'uploaded_at': datetime.now().isoformat(),
                    'method': 'manual'
                }
            })
        
        return items
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for file processor"""
        return {
            'pypdf2_available': 'PyPDF2' in globals(),
            'python_docx_available': 'python-docx' in globals(),
            'status': 'ready'
        }
