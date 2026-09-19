"""
Script to define the data schemas for the router model.
The router model classifies queries into two categories:
1. SIMPLE: Direct, fact-based queries that can be answered by the hybrid RAG system (BM25 + Dense).
2. COMPLEX: Queries that require multi-step reasoning or involve comparisons, negations...
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"       # Traited by direct hybrid-RAG (BM25 + Dense)
    COMPLEX = "COMPLEXE"    # Traited by agent (ReAct / Multi-step)

class RouterDecision(BaseModel):
    query: str
    complexity: QueryComplexity

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, 
        description="Confidence score in decision (1.0 = certain, 0.0 = uncertain)"
    )

    decision_source: str = Field(description="The source of the decision: 'HEURISTIC' or 'CLASSIFIER_MODEL'")

    detected_keywords: List[str] = Field(default_factory=list, description="Logical keywords that triggered the decision")

    reasoning: str = Field(description= "Explanation of the routing decision, including heuristics or model-based reasoning.")