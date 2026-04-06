"""
Problem data model
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class Problem(BaseModel):
    """Problem model representing extracted user problems"""
    
    id: Optional[str] = None
    problem_statement: str = Field(..., description="Clear problem statement")
    problem_category: str = Field(..., description="Category of problem (ui, feature, pricing, support, performance)")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    frequency: Dict[str, Any] = Field(default_factory=dict, description="Frequency information")
    user_segments: List[str] = Field(default_factory=list, description="Affected user segments")
    source_mix: Dict[str, float] = Field(default_factory=dict, description="Source distribution")
    potential_impact: Dict[str, float] = Field(default_factory=dict, description="Business impact assessment")
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in problem")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence")
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
    def from_dict(cls, data: Dict[str, Any]) -> "Problem":
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        return cls(**data)


class ProblemCreate(BaseModel):
    """Problem creation model"""
    
    problem_statement: str
    problem_category: str
    severity: str
    frequency: Dict[str, Any] = Field(default_factory=dict)
    user_segments: List[str] = Field(default_factory=list)
    source_mix: Dict[str, float] = Field(default_factory=dict)
    potential_impact: Dict[str, float] = Field(default_factory=dict)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class ProblemUpdate(BaseModel):
    """Problem update model"""
    
    problem_statement: Optional[str] = None
    problem_category: Optional[str] = None
    severity: Optional[str] = None
    frequency: Optional[Dict[str, Any]] = None
    user_segments: Optional[List[str]] = None
    source_mix: Optional[Dict[str, float]] = None
    potential_impact: Optional[Dict[str, float]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[List[Dict[str, Any]]] = None


class ProblemFilter(BaseModel):
    """Problem filter model for queries"""
    
    problem_category: Optional[str] = None
    severity: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_segments: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
