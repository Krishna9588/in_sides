"""
Database configuration and connection management
"""
from supabase import create_client, Client
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

from .settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.client: Client = None
        self._connect()
    
    def _connect(self):
        """Initialize database connection"""
        try:
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info("Connected to Supabase database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def get_client(self) -> Client:
        """Get database client"""
        if not self.client:
            self._connect()
        return self.client
    
    def health_check(self) -> bool:
        """Check database health"""
        try:
            result = self.client.table('signals').select('count').execute()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


class BaseRepository:
    """Base repository for database operations"""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.client = db_manager.get_client()
    
    async def create(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new record"""
        try:
            # Add timestamps
            data['created_at'] = datetime.now().isoformat()
            data['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table(self.table_name).insert(data).execute()
            
            if result.data:
                logger.info(f"Created record in {self.table_name}: {result.data[0]['id']}")
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to create record in {self.table_name}: {e}")
            return None
    
    async def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get record by ID"""
        try:
            result = self.client.table(self.table_name).select('*').eq('id', record_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get record {record_id} from {self.table_name}: {e}")
            return None
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all records with pagination"""
        try:
            result = self.client.table(self.table_name).select('*').range(offset, offset + limit - 1).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get records from {self.table_name}: {e}")
            return []
    
    async def update(self, record_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a record"""
        try:
            # Add updated timestamp
            data['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table(self.table_name).update(data).eq('id', record_id).execute()
            
            if result.data:
                logger.info(f"Updated record {record_id} in {self.table_name}")
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to update record {record_id} in {self.table_name}: {e}")
            return None
    
    async def delete(self, record_id: str) -> bool:
        """Delete a record"""
        try:
            result = self.client.table(self.table_name).delete().eq('id', record_id).execute()
            
            if result.data:
                logger.info(f"Deleted record {record_id} from {self.table_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete record {record_id} from {self.table_name}: {e}")
            return False
    
    async def query(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query records with filters"""
        try:
            query = self.client.table(self.table_name).select('*')
            
            # Apply filters
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle complex filters like {'gte': value}
                    for operator, val in value.items():
                        if operator == 'gte':
                            query = query.gte(key, val)
                        elif operator == 'lte':
                            query = query.lte(key, val)
                        elif operator == 'like':
                            query = query.like(key, val)
                else:
                    query = query.eq(key, value)
            
            query = query.limit(limit)
            result = query.execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to query {self.table_name}: {e}")
            return []


class SignalRepository(BaseRepository):
    """Repository for signals"""
    
    def __init__(self):
        super().__init__('signals')
    
    async def get_recent_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent signals ordered by created_at"""
        try:
            result = self.client.table('signals').select('*').order('created_at', desc=True).limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent signals: {e}")
            return []
    
    async def get_signals_by_source(self, source_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get signals by source type"""
        return await self.query({'source_type': source_type}, limit)
    
    async def search_signals(self, query_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search signals by content"""
        try:
            result = self.client.table('signals').select('*').ilike('content', f'%{query_text}%').limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to search signals: {e}")
            return []


class ProblemRepository(BaseRepository):
    """Repository for problems"""
    
    def __init__(self):
        super().__init__('problems')
    
    async def get_recent_problems(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent problems ordered by created_at"""
        try:
            result = self.client.table('problems').select('*').order('created_at', desc=True).limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent problems: {e}")
            return []
    
    async def get_problems_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get problems by category"""
        return await self.query({'problem_category': category}, limit)


class InsightRepository(BaseRepository):
    """Repository for insights"""
    
    def __init__(self):
        super().__init__('insights')
    
    async def get_recent_insights(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent insights ordered by created_at"""
        try:
            result = self.client.table('insights').select('*').order('created_at', desc=True).limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent insights: {e}")
            return []
    
    async def get_insights_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get insights by category"""
        return await self.query({'insight_category': category}, limit)


class BriefRepository(BaseRepository):
    """Repository for product briefs"""
    
    def __init__(self):
        super().__init__('product_briefs')
    
    async def get_recent_briefs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent briefs ordered by created_at"""
        try:
            result = self.client.table('product_briefs').select('*').order('created_at', desc=True).limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent briefs: {e}")
            return []


class ConversationRepository(BaseRepository):
    """Repository for conversations"""
    
    def __init__(self):
        super().__init__('conversations')
    
    async def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        return await self.get_by_id(conversation_id)
    
    async def create_conversation(self, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Create new conversation"""
        conversation_data = {
            'user_id': user_id,
            'status': 'active',
            'metadata': {}
        }
        return await self.create(conversation_data)
    
    async def update_conversation(self, conversation_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update conversation"""
        return await self.update(conversation_id, updates)


class MessageRepository(BaseRepository):
    """Repository for conversation messages"""
    
    def __init__(self):
        super().__init__('messages')
    
    async def get_messages_by_conversation(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a conversation"""
        return await self.query({'conversation_id': conversation_id}, limit)
    
    async def create_message(self, conversation_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Create new message"""
        message_data = {
            'conversation_id': conversation_id,
            'message_type': message_type,  # 'user' or 'assistant'
            'content': content,
            'metadata': metadata or {}
        }
        return await self.create(message_data)


# Repository instances
signal_repo = SignalRepository()
problem_repo = ProblemRepository()
insight_repo = InsightRepository()
brief_repo = BriefRepository()
conversation_repo = ConversationRepository()
message_repo = MessageRepository()
