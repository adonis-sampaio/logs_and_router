from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class LogEntry(BaseModel):
    timestamp: str
    source: str
    level: str
    message: str
    metadata: Optional[Dict] = None


class CausalNode(BaseModel):
    event_id: str
    timestamp: str
    description: str
    nodetype: str
    confidence: float
    category: Optional[str] = None


class CausalChain(BaseModel):
    nodes: List[CausalNode]
    edges: List[Dict]
    root_cause: Optional[CausalNode] = None


class AnalysisRequest(BaseModel):
    logs: List[LogEntry]
    context: Optional[str] = None
    llm_model: Optional[str] = None


class SymptomRequest(BaseModel):
    log_text: str
    context: Optional[str] = None


class RootCauseSuggestion(BaseModel):
    category: str
    score: float
    reason: str
    evidence: List[str] = []


class RootCauseResponse(BaseModel):
    likely_root_cause: str
    confidence: float
    reason: str
    possible_causes: List[RootCauseSuggestion]
    evidence: List[str] = []
    debug: Dict[str, List[str]] = {}


class AnalysisResponse(BaseModel):
    model_used: str
    root_cause_category: str
    causal_chain: Dict
    routing: Dict
    explanation: str