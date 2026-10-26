"""
Script to define the data schemas for the router model.
The router model classifies queries into three categories:

0. SIMPLE: Direct, fact-based queries that can be answered by the hybrid RAG system (BM25 + Dense).
1. COMPLEX: Queries that require multi-step reasoning or involve comparisons, negations...
2. HORS_CONTEXT: Queries that are greetings or unrelated to the domain of agricultural regulations, answered bu LLM without augmenting the context.
The schema is defined using Pydantic models, which provide data validation and serialization.
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"       # Traited by direct hybrid-RAG (BM25 + Dense)
    COMPLEX = "COMPLEXE"    # Traited by agent (sub_queries)
    HORS_CONTEXT = "HORS_CONTEXT"  # Traited by LLM without context augmentation

class RouterDecision(BaseModel):
    query: str
    complexity: QueryComplexity

    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score in decision (1.0 = certain, 0.0 = uncertain)")

    decision_source: str = Field(description="The source of the decision: 'HEURISTIC' or 'CLASSIFIER_MODEL'")

    detected_keywords: List[str] = Field(default_factory=list, description="Logical keywords that triggered the decision")

    reasoning: str = Field(description= "Explanation of the routing decision, including heuristics or model-based reasoning.")