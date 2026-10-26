"""
Script to implement the full orchestrator using Groq Cloud API.
- The orchestrator orchestrates answering user queries.
- Intent Router -> Routing decision.
- Simple RAG or Reasoning Agent -> Context formatting & Generation via Groq.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq

from src.Cache.semantic_cache import SemanticCache
from src.retrieval.hybrid_search import HybridRetriever
from src.router.router import IntentRouter
from src.agent.reasoning_agent import ReasoningAgent
from src.router.schemas import QueryComplexity

SYSTEM_PROMPT = """Tu es un assistant expert de la réglementation phytosanitaire française (données E-Phy).
Ton rôle est de répondre aux questions des utilisateurs en t'appuyant STRICTEMENT sur les documents fournis en contexte.

Règles génerales:
1. Si l'information exacte n'est pas présente dans les documents fournis, réponds explicitement : "Je ne dispose pas de cette information."
2. Ne suppose jamais de valeurs et n'invente aucun délai, dosage ou numéro d'AMM.
3. Cite le nom du produit et son numéro d'AMM dans ta réponse lorsque c'est pertinent.
4. Reste direct, précis et factuel.

Règles impératives de traçabilité et de rigueur :
1. Chaque affirmation, chiffre, dose ou statut d'autorisation doit obligatoirement citer sa source entre crochets à la fin de la phrase ou dans le tableau (ex. [SOURCE 1], [SOURCE 2]).
2. Pour chaque étape de raisonnement ou point de comparaison, mentionne explicitement le produit, l'AMM et la source documentaire utilisée.
4. Conclus systématiquement par une section récapitulative listant la correspondance exacte entre chaque [SOURCE X] et la fiche E-Phy associée (Nom, AMM, Culture).
"""

class Orchestrator:
    def __init__(self, cache = None, retriever=None, router=None, 
        model_simple: str = "openai/gpt-oss-20b", model_reasoning: str = "openai/gpt-oss-120b"
    ):
        self.cache = cache or SemanticCache()
        self.retriever = retriever or HybridRetriever()
        self.router = router or IntentRouter()
        self.model_simple = model_simple
        self.model_reasoning = model_reasoning

        # Récupération de la clé API (gère os.environ et st.secrets)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError("GROQ_API_KEY introuvable. Définissez-la dans les variables d'environnement ou les secrets Streamlit.")

        self.client = Groq(api_key=api_key)
        self.reasoning_agent = ReasoningAgent(retriever=self.retriever, model_name=self.model_reasoning)

    
    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Formate le contexte documentaire avec des identifiants explicites."""
        if not docs:
            return "Aucun document trouvé."

        formatted_chunks = []
        for idx, doc in enumerate(docs, start=1):
            formatted_chunks.append(
                f"--- [SOURCE {idx}] ---\n"
                f"Produit : {doc.get('nom_produit', 'Inconnu')}\n"
                f"AMM : {doc.get('numero_amm', 'N/A')}\n"
                f"Culture : {doc.get('culture', 'N/A')}\n"
                f"Type d'usage : {doc.get('type_usage', 'N/A')}\n"
                f"Contenu réglementaire :\n{doc.get('content', '')}\n"
            )
        return "\n".join(formatted_chunks)


 
    def answer(self, query: str, top_k: int = 3, culture: Optional[str] = None, numero_amm=None, type_usage=None, use_llm: bool = True):
        """Pipeline: (Embediing de query -> Matching with Cache) -> Routing -> Retrieval -> Formatting -> Inférence Groq."""

        # we check if query matches a known intent in the cache (greetings, out_of_scope, etc.)(the cache is in semantic_cache_seeds.json)
        cache_response = self.cache.match(query)
        if cache_response:
            intent, response, score = cache_response
            print(f"[CACHE HIT] Intent: {intent} | Score: {score:.4f}\nRéponse: {response}")
            return {"answer": response, "sources": [], "query_complexity": "HORS_CONTEXT"}
        
        print("[CACHE MISS] Requête à transmettre au routeur d'intention.")
        
        # Routing d'intention
        intent_router_decision = self.router.route(query=query)
        is_complex = getattr(intent_router_decision, "complexity", None) == QueryComplexity.COMPLEX
        is_hors_context = getattr(intent_router_decision, "complexity", None) == QueryComplexity.HORS_CONTEXT
        
        
        if is_complex:
            print(f"[RAG-Enhanced-Agent] Passage via ReasoningAgent ({self.reasoning_agent.model_name})...")
            result = self.reasoning_agent.run(query=query, top_k_per_subquery=top_k)

            return {
                "answer": result.get("response", ""),
                "sources": result.get("sources", []) ,
                "sub_queries": result.get("sub_queries", []),
                "Thought_Decomposition": result.get("thought_decomposition", ""),
                "query_complexity": "COMPLEX",
            }
        
        else: # Either simple query or hors_context query
            if is_hors_context:
                print(f"[LLM] Passage via LLM without augmented context ({self.model_simple})...")
                # we use llm without augmented context  
                result = ""
                retrieved_docs = []

            else:
                print(f"[RAGAgent] Aiguillage RAG Simple ({self.model_simple})...")
                retrieved_docs = self.retriever.search(query=query, top_k=top_k, culture=culture, numero_amm=numero_amm, type_usage=type_usage)

            # Format context for both simple and hors_context queries
            context_str = self._format_context(retrieved_docs)
            user_prompt = f"CONTEXTE:\n{context_str}\n\nQUESTION:\n{query}"

            response_text = ""
            if use_llm:
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=self.model_simple,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1, max_tokens=1024
                    )
                
                    response_text = chat_completion.choices[0].message.content
                except Exception as e:
                    response_text = f"[Erreur Groq] Échec de l'inférence : {e}"
            else:
                response_text = "[Info] LLM désactivé. Le contexte a été préparé avec succès."

            return {
                "answer": response_text,
                "sources": retrieved_docs,
                "query_complexity": "HORS_CONTEXT" if is_hors_context else "SIMPLE",
            }



    def close(self):
        if self.retriever:
            self.retriever.close()