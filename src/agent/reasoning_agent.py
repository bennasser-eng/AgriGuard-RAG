"""
Script to implement the Reasoning Agent for complex queries.
- Decomposes complex user queries into sub-queries.
- Executes sub-queries via HybridRetriever.
- Synthesizes findings using a larger/stronger LLM.
"""

import json
from typing import List, Dict, Any
import ollama
from src.retrieval.hybrid_search import HybridRetriever


DECOMPOSITION_PROMPT = """Tu es un planificateur d'extraction d'information pour la réglementation phytosanitaire (E-Phy).
Ta tâche est de décomposer une question complexe en un ensemble de 2 à 3 sous-questions simples et indépendantes qui permettront de chercher dans une base vectorielle.

Règles :
1. Chaque sous-question doit chercher UNE SEULE information précise (ex: dosage, culture, délai, produit).
2. Réponds UNIQUEMENT sous forme d'un objet JSON strict respectant ce schéma exact, sans aucun texte autour :

{
  "sub_queries": [
    "sous-question 1",
    "sous-question 2"
  ]
}
"""

SYNTHESIS_PROMPT = """Tu es un assistant expert de la réglementation phytosanitaire française (données E-Phy).
Ton rôle est de répondre à une question complexe en analysant et combinant l'ensemble des documents fournis en contexte.

Règles de conduite :
1. Si l'information exacte pour répondre à la question globale ou à une des parties n'est pas présente dans les documents fournis, réponds explicitement ce qui manque.
2. Ne suppose jamais de valeurs et n'invente aucun délai, dosage ou numéro d'AMM.
3. Cite systématiquement le nom des produits et leurs numéros d'AMM.
4. Explique clairement ton raisonnement étape par étape dans la réponse finale.
"""


class ReasoningAgent:
    def __init__(self, retriever = None, model_name: str = "qwen2.5:7b"):
        self.retriever = retriever or HybridRetriever()
        self.model_name = model_name

    def _decompose_query(self, query: str) -> List[str]:
        """Décompose one complex query into multiple simpler ones."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": DECOMPOSITION_PROMPT},
                    {"role": "user", "content": f"Question complexe : {query}"}
                ],
                format="json",     # Force the mode JSON under Ollama
                keep_alive=0  # Décharge le modèle immédiatement
            )
            data = json.loads(response["message"]["content"])
            return data.get("sub_queries", [query])
        except Exception as e:
            print(f"[ReasoningAgent Warning] Échec de décomposition : {e}. Utilisation de la question brute.")
            return [query]


    def _deduplicate_docs(self, docs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dédoublonne documents retrieved by different sub-queries."""
        seen_ids = set()
        unique_docs = []
        for doc in docs_list:
            # We use the Qdrant ID or a unique signature
            doc_id = doc.get("id") or f"{doc.get('numero_amm')}_{doc.get('culture')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        return unique_docs


    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format the entire set of retrieved documents for the final prompt."""
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


    def run(self, query: str, top_k_per_subquery: int = 2) -> Dict[str, Any]:
        """Exécute the complex reasoning pipeline."""
        
        # Phase 1 : Décomposition of the query
        sub_queries = self._decompose_query(query)
        
        # Phase 2 : Extraction multi-requests
        all_retrieved_docs = []
        for sub_q in sub_queries:
            docs = self.retriever.search(query=sub_q, top_k=top_k_per_subquery)
            all_retrieved_docs.extend(docs)

        # Phase 3 : Dédoublonnage & Formatage
        unique_docs = self._deduplicate_docs(all_retrieved_docs)
        context_str = self._format_context(unique_docs)
        
        user_prompt = f"CONTEXTE AGRÉGÉ :\n{context_str}\n\nQUESTION COMPLEXE :\n{query}"

        # Phase 4 : Synthèse par le LLM puissant
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_text = response["message"]["content"]
        except Exception as e:
            response_text = f"[Erreur ReasoningAgent] Échec de la génération : {e}"

        return {
            "query": query,
            "sub_queries": sub_queries,
            "response": response_text,
            "context_used": unique_docs,
            "formatted_prompt": user_prompt
        }


    def close(self):
        if hasattr(self, "retriever") and self.retriever:
            self.retriever.close()