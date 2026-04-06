"""
Product Brief data model
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class ProductBrief(BaseModel):
    """Product Brief model representing actionable product recommendations"""
    
    id: Optional[str] = None
    brief_title: str = Field(..., description="Title of the product brief")
    feature_name: str = Field(..., description="Name of the recommended feature")
    problem_statement: Dict[str, Any] = Field(..., description="Problem statement and analysis")
    opportunity_assessment: Dict[str, Any] = Field(..., description="Market opportunity analysis")
    solution_design: Dict[str, Any] = Field(..., description="Solution design details")
    impact_assessment: Dict[str, Any] = Field(..., description="Impact and ROI assessment")
    implementation_plan: Dict[str, Any] = Field(..., description="Implementation plan")
    success_metrics: Dict[str, Any] = Field(..., description="Success metrics and KPIs")
    prioritization_score: Dict[str, Any] = Field(..., description="Prioritization score and recommendation")
    validation_status: str = Field(default="pending", description="Validation status")
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
    def from_dict(cls, data: Dict[str, Any]) -> "ProductBrief":
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        return cls(**data)


class ProductBriefCreate(BaseModel):
    """Product Brief creation model"""
    
    brief_title: str
    feature_name: str
    problem_statement: Dict[str, Any]
    opportunity_assessment: Dict[str, Any]
    solution_design: Dict[str, Any]
    impact_assessment: Dict[str, Any]
    implementation_plan: Dict[str, Any]
    success_metrics: Dict[str, Any]
    prioritization_score: Dict[str, Any]
    validation_status: str = Field(default="pending")


class ProductBriefUpdate(BaseModel):
    """Product Brief update model"""
    
    brief_title: Optional[str] = None
    feature_name: Optional[str] = None
    problem_statement: Optional[Dict[str, Any]] = None
    opportunity_assessment: Optional[Dict[str, Any]] = None
    solution_design: Optional[Dict[str, Any]] = None
    impact_assessment: Optional[Dict[str, Any]] = None
    implementation_plan: Optional[Dict[str, Any]] = None
    success_metrics: Optional[Dict[str, Any]] = None
    prioritization_score: Optional[Dict[str, Any]] = None
    validation_status: Optional[str] = None


class ProductBriefFilter(BaseModel):
    """Product Brief filter model for queries"""
    
    validation_status: Optional[str] = None
    min_priority_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
