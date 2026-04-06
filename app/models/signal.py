"""
Signal data model
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class Signal(BaseModel):
    """Signal model representing collected data"""
    
    id: Optional[str] = None
    source_type: str = Field(..., description="Type of data source")
    entity: str = Field(..., description="Source entity name")
    signal_type: str = Field(..., description="Type of signal (complaint, feature, trend, insight)")
    content: str = Field(..., description="Signal content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in signal")
    relevance_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Relevance score")
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
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        return cls(**data)


class SignalCreate(BaseModel):
    """Signal creation model"""
    
    source_type: str
    entity: str
    signal_type: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.8, ge=0.0, le=1.0)


class SignalUpdate(BaseModel):
    """Signal update model"""
    
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class SignalFilter(BaseModel):
    """Signal filter model for queries"""
    
    source_type: Optional[str] = None
    signal_type: Optional[str] = None
    entity: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
