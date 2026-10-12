"""
Script to implement the full orchestrator using Groq Cloud API.
- The orchestrator orchestrates answering user queries.
- Intent Router -> Routing decision.
- Simple RAG or Reasoning Agent -> Context formatting & Generation via Groq.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq

from src.retrieval.hybrid_search import HybridRetriever
from src.router.router import IntentRouter
from src.agent.reasoning_agent import ReasoningAgent
from src.router.schemas import QueryComplexity

SYSTEM_PROMPT = """Tu es un assistant expert de la réglementation phytosanitaire française (données E-Phy).
Ton rôle est de répondre aux questions des utilisateurs en t'appuyant STRICTEMENT sur les documents fournis en contexte.

Règles de conduite :
1. Si l'information exacte n'est pas présente dans les documents fournis, réponds explicitement : "Je ne dispose pas de cette information dans la base de données E-Phy."
2. Ne suppose jamais de valeurs et n'invente aucun délai, dosage ou numéro d'AMM.
3. Cite le nom du produit et son numéro d'AMM dans ta réponse lorsque c'est pertinent.
4. Reste direct, précis et factuel.
"""

class Orchestrator:
    def __init__(
        self, 
        retriever=None, 
        router=None, 
        model_simple: str = "openai/gpt-oss-20b", 
        model_reasoning: str = "openai/gpt-oss-120b"
    ):
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
        """Formate le contexte documentaire récupéré."""
        if not docs:
            return "Aucun document trouvé."

        formatted_chunks = []
        for doc in docs:
            formatted_chunks.append(
                f"Nom produit: {doc.get('nom_produit', '')}\n"
                f"AMM: {doc.get('numero_amm', '')}\n"
                f"Culture: {doc.get('culture', '')}\n"
                f"Type usage: {doc.get('type_usage', '')}\n"
                f"Contenu:\n{doc.get('content', '')}\n"
            )
        return "\n".join(formatted_chunks)

    def answer(self, query: str, top_k: int = 3, culture: Optional[str] = None, numero_amm=None, type_usage=None, use_llm: bool = True):
        """Pipeline complet : Routing -> Retrieval -> Formatting -> Inférence Groq."""
        
        # 1. Routing d'intention
        intent_router_decision = self.router.route(query=query)
        is_complex = getattr(intent_router_decision, "complexity", None) == QueryComplexity.COMPLEX

        if is_complex:
            print(f"[RAG-Enhanced-Agent] Passage via ReasoningAgent ({self.reasoning_agent.model_name})...")
            result = self.reasoning_agent.run(query=query, top_k_per_subquery=top_k)
            result["query_complexity"] = intent_router_decision.complexity
            return result
        else:
            print(f"[RAGAgent] Aiguillage RAG Simple ({self.model_simple})...")
            # 2. Recherche hybride des fiches E-Phy
            retrieved_docs = self.retriever.search(
                query=query, 
                top_k=top_k, 
                culture=culture, 
                numero_amm=numero_amm, 
                type_usage=type_usage
            )

            # 3. Formatage du contexte
            context_str = self._format_context(retrieved_docs)
            user_prompt = f"CONTEXTE:\n{context_str}\n\nQUESTION:\n{query}"

            # 4. Génération via API Groq
            response_text = ""
            if use_llm:
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=self.model_simple,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1024
                    )
                    response_text = chat_completion.choices[0].message.content
                except Exception as e:
                    response_text = f"[Erreur Groq] Échec de l'inférence : {e}"
            else:
                response_text = "[Info] LLM désactivé. Le contexte a été préparé avec succès."

            return {
                "query": query,
                "response": response_text,
                "formatted_prompt": user_prompt,
                "query_complexity": getattr(intent_router_decision, "complexity", None),
                "intent_router_decision": intent_router_decision,
                "context_used": retrieved_docs
            }

    def close(self):
        if self.retriever:
            self.retriever.close()