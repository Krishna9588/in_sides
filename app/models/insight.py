"""
Insight data model
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class Insight(BaseModel):
    """Insight model representing synthesized insights"""
    
    id: Optional[str] = None
    insight_title: str = Field(..., description="Title of the insight")
    insight_statement: str = Field(..., description="Clear insight statement")
    insight_category: str = Field(..., description="Category of insight")
    supporting_problems: List[str] = Field(default_factory=list, description="IDs of supporting problems")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
    root_causes: List[Dict[str, Any]] = Field(default_factory=list, description="Identified root causes")
    implications: Dict[str, Any] = Field(default_factory=dict, description="Business implications")
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in insight")
    strategic_importance: str = Field(..., description="Strategic importance (high, medium, low)")
    urgency: str = Field(..., description="Urgency level (immediate, short_term, long_term)")
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
    def from_dict(cls, data: Dict[str, Any]) -> "Insight":
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        return cls(**data)


class InsightCreate(BaseModel):
    """Insight creation model"""
    
    insight_title: str
    insight_statement: str
    insight_category: str
    supporting_problems: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    root_causes: List[Dict[str, Any]] = Field(default_factory=list)
    implications: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    strategic_importance: str
    urgency: str


class InsightUpdate(BaseModel):
    """Insight update model"""
    
    insight_title: Optional[str] = None
    insight_statement: Optional[str] = None
    insight_category: Optional[str] = None
    supporting_problems: Optional[List[str]] = None
    evidence: Optional[Dict[str, Any]] = None
    root_causes: Optional[List[Dict[str, Any]]] = None
    implications: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    strategic_importance: Optional[str] = None
    urgency: Optional[str] = None


class InsightFilter(BaseModel):
    """Insight filter model for queries"""
    
    insight_category: Optional[str] = None
    strategic_importance: Optional[str] = None
    urgency: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
