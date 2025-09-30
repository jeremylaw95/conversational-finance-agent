"""Response models for agent interactions."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ResponseStatus(str, Enum):
    """Status of agent response."""
    
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class CalculationStep(BaseModel):
    """Individual calculation step with validation."""
    
    operation: str = Field(description="The operation performed")
    operands: List[Union[float, str]] = Field(description="Input values")
    result: Union[float, str] = Field(description="Calculation result")
    step_number: int = Field(description="Order in calculation sequence")


class ValidationResult(BaseModel):
    """Result of response validation."""
    
    is_valid: bool
    format_correct: bool
    has_reasoning: bool
    has_answer: bool
    error_messages: List[str] = Field(default_factory=list)


class EnhancedAgentResponse(AgentResponse):
    """Extended agent response with additional metadata."""
    
    status: ResponseStatus = ResponseStatus.SUCCESS
    calculation_steps: List[CalculationStep] = Field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    context_references: List[str] = Field(default_factory=list)
    error_details: Optional[str] = None

