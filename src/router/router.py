"""
Script for routing of intents to determine if a query is SIMPLE or COMPLEX or HORS_CONTEXT.
- Simple queries are handled by direct Hybrid RAG (BM25 + Dense).
- Complex queries are handled by a Reasoning Agent (which divide query into sub-queries and execute them sequentially using RAG).
- Hors_context queries are handled by LLM without context augmentation
Routing is based on a combination of heuristic rules and an ML classifier.
"""

import re
import os
import joblib
from typing import Optional
from .schemas import QueryComplexity, RouterDecision

class IntentRouter:

    COMPLEX_KEYWORDS = [
        # Verbes et variantes de comparaison
        "comparer", "compare", "comparatif", "comparaison",
        "différence", "différent", "différentes",
        "la même", "le même", "les mêmes", "identique",
        "versus", "vs", "choisir entre",
        # Notions agronomiques complexes
        "programme", "mélange", "compatibilité", "alterner", "alternance",
        "sans", "alternative", "compromis", "efficacité relative"
    ]

    HORS_CONTEXT_KEYWORDS = ["finance", "meteo", "politique", "sport", "football", 
                            "basketball", "hockey", "handball", "cinema", "musique",
                            "voyage", "tourisme", "économie", "cours du marché"]
    

    # Regex couvrant les symboles (<, >) ET les formulations textuelles (inférieur à, moins de...)
    CONSTRAINT_REGEX = r'(?:<|>|<=|>=|=|inférieur à|supérieur à|moins de|plus de|strictement inférieur|strictement supérieur)\s*\d+(?:[\.,]\d+)?\s*(?:j|jours|m|mètres|l/ha|kg/ha|g/ha)?'

    def __init__(self, model_path: Optional[str] = "models/router_model.joblib"):
        self.model_path = model_path
        self.classifier_pipeline = None

        # load the model_path
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.classifier_pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"[WARN] Not possible to load the routing model ({e}). Only heuristic mode activated.")


    def _check_heuristics(self, query: str) -> Optional[RouterDecision]:
        """Fast check by logical rules (< 1 ms)."""
        query_lower = query.lower()

        # Check if the query contains any hors-context keywords
        if any(kw in query_lower for kw in self.HORS_CONTEXT_KEYWORDS):
            return RouterDecision(query=query, complexity=QueryComplexity.HORS_CONTEXT,
                confidence=1.0, decision_source="HEURISTIC",
                detected_keywords=[kw for kw in self.HORS_CONTEXT_KEYWORDS if kw in query_lower],
                reasoning="Routing reason is: Query contains hors-context keywords."
            )
        
        # Detertmine if the query contains some complex keywords
        matched_keywords = [kw for kw in self.COMPLEX_KEYWORDS if kw in query_lower]
        
        # Determine if the query contains any inequality constraints (e.g., DAR < 30j, ZNT <= 5m, dose > 1.5 L/ha)
        matched_constraints = re.findall(self.CONSTRAINT_REGEX, query_lower)

        # If the query contains complex keywords or multiple inequality constraints -> COMPLEX
        if matched_keywords or len(matched_constraints) >= 1:
            reason = []
            if matched_keywords:
                reason.append(f"Key words found: {matched_keywords}")
            if len(matched_constraints) >= 2:
                reason.append(f"{len(matched_constraints)} numerical constraints detected: {matched_constraints}")

            return RouterDecision(query=query, complexity=QueryComplexity.COMPLEX, confidence=1.0, 
                                  decision_source="HEURISTIC", detected_keywords=matched_keywords, 
                                  reasoning="Routing reason is: (" + " et ".join(reason) + ")."
            )

        return None
    

    def route(self, query: str) -> RouterDecision:
        """The principal routing function to determine if a query is SIMPLE, COMPLEX or HORS_CONTEXT."""
        # Phase 1: Filter heuristically first (fast, < 1 ms)
        heuristic_decision = self._check_heuristics(query)
        if heuristic_decision is not None:
            return heuristic_decision

        # Phase 2: Classification by ML model (model tres leger)
        if self.classifier_pipeline is not None:
            # (0 = SIMPLE, 1 = COMPLEX, 2 = HORS_CONTEXT)
            pred = self.classifier_pipeline.predict([query])[0]
            
            # Probabilities for the confidence
            proba = self.classifier_pipeline.predict_proba([query])[0]
            confidence = float(proba[pred])

            if pred == 2:
                complexity = QueryComplexity.HORS_CONTEXT
            elif pred == 1:
                complexity = QueryComplexity.COMPLEX
            elif pred == 0:
                complexity = QueryComplexity.SIMPLE
            else:
                raise ValueError(f"Unexpected prediction value: {pred}")
            
            return RouterDecision(query=query, complexity=complexity, confidence=round(confidence, 4), decision_source="CLASSIFIER_MODEL", 
                                  reasoning=f"Routing {complexity.value} based on the ML model (confidence: {confidence:.2%})."
            )

        # Phase 3: Fallback by default to HORS_CONTEXT if no model is loaded
        return RouterDecision(query=query, complexity=QueryComplexity.HORS_CONTEXT, confidence=0.5,
            decision_source="FALLBACK_DEFAULT", reasoning="Routing HORS_CONTEXT by default (no ML model loaded)."
        )