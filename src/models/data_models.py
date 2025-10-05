"""Data models matching the ConvFinQA dataset structure."""

from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Financial document structure from dataset."""
    
    pre_text: str = Field(description="Text before the table")
    post_text: str = Field(description="Text after the table")
    table: Dict[str, Dict[str, Union[float, str, int]]] = Field(
        description="Financial data table with periods as keys"
    )


class Dialogue(BaseModel):
    """Conversation dialogue structure matching dataset."""
    
    conv_questions: List[str] = Field(description="Questions in conversation")
    conv_answers: List[str] = Field(description="Expected answers")
    turn_program: List[str] = Field(description="DSL programs for each turn")
    executed_answers: List[Union[float, str]] = Field(description="Program execution results")
    qa_split: List[bool] = Field(description="Question source indicators")


class Features(BaseModel):
    """Dataset features for analysis."""
    
    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    """Complete dataset record matching actual JSON structure."""
    
    id: str
    doc: Document
    dialogue: Dialogue  # This matches the actual dataset key
    features: Features


class AgentResponse(BaseModel):
    """Agent's response to a question."""
    
    answer: str
    reasoning_steps: List[str]
    confidence: float
    raw_response: str
    processing_time: float


class ConversationTurn(BaseModel):
    """Individual turn in a conversation."""
    
    question: str
    answer: str
    turn_number: int
    response_details: Optional[AgentResponse] = None

