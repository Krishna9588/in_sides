"""
Conversation data models
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class Conversation(BaseModel):
    """Conversation model for chat interactions"""
    
    id: Optional[str] = None
    user_id: Optional[str] = None
    status: str = Field(default="active", description="Conversation status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        return cls(**data)


class Message(BaseModel):
    """Message model for conversation messages"""
    
    id: Optional[str] = None
    conversation_id: str = Field(..., description="ID of the conversation")
    message_type: str = Field(..., description="Type of message (user, assistant)")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        return cls(**data)


class ConversationCreate(BaseModel):
    """Conversation creation model"""
    
    user_id: Optional[str] = None
    status: str = Field(default="active")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageCreate(BaseModel):
    """Message creation model"""
    
    conversation_id: str
    message_type: str = Field(..., regex="^(user|assistant)$")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Query request model for chat"""
    
    query: str = Field(..., description="User query")
    conversation_id: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Query response model"""
    
    response_id: str
    query: str
    answer: Dict[str, Any]
    evidence: Dict[str, Any]
    confidence: Dict[str, Any]
    metadata: Dict[str, Any]
    follow_up_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
