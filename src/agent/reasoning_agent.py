"""
Script to implement the Reasoning Agent for complex queries using Groq Cloud API.
- Decomposes complex user queries into sub-queries.
- Executes sub-queries via HybridRetriever.
- Synthesizes findings using a stronger LLM (e.g., Llama 3.3 70B).
"""

import os
import json
from typing import List, Dict, Any, Tuple
from groq import Groq
from src.retrieval.hybrid_search import HybridRetriever

DECOMPOSITION_PROMPT = """Tu es un planificateur d'extraction d'information pour la réglementation phytosanitaire (E-Phy).
Ta tâche est de décomposer une question complexe en un ensemble de 2 à 3 sous-questions simples et indépendantes qui permettront de chercher dans une base vectorielle.

Règles impératives :
1. "thought" : 1 à 2 phrases directes résumant la vérification réglementaire ciblée (produits, cultures, contraintes).
2. "sub_queries" : 2 à 3 sous-requêtes techniques précises (cherchant AMM, usages, doses, DAR ou ZNT).
3. Réponds UNIQUEMENT via un objet JSON strict respectant ce schéma exact :

{
  "thought": "Vérification des autorisations et comparaison des doses/DAR pour les spécialités mentionnées sur blé.",
  "sub_queries": [
    "usages et dose autorisée produit A sur blé",
    "usages et dose autorisée produit B sur blé"
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
    def __init__(self, retriever=None, model_name=None):
        if not model_name:
            raise ValueError("ReasoningAgent requiert un 'model_name' valide fourni par l'Orchestrateur.")
        self.retriever = retriever or HybridRetriever()
        self.model_name = model_name
        self.thought_decomposition = ""

        # la clé API Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError("GROQ_API_KEY introuvable pour ReasoningAgent.")

        self.client = Groq(api_key=api_key)



    def _decompose_query(self, query: str) -> Tuple[List[str], str]:
        """Décompose une question complexe en sous-questions simples via JSON mode, et envoie d'un thought à afficher"""
        try:
            response = self.client.chat.completions.create( 
                model=self.model_name,   
                messages=[
                    {"role": "system", "content": DECOMPOSITION_PROMPT},
                    {"role": "user", "content": f"Question complexe : {query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)

            sub_queries = data.get("sub_queries", [query])
            thought = data.get("thought", "Analyse réglementaire multi-critères.")
            
            # Sécurité si le LLM a renvoyé une liste pour thought
            if isinstance(thought, list):
                thought = " ".join(thought)

            return sub_queries, thought

        except Exception as e:
            print(f"[ReasoningAgent Warning] Échec de décomposition : {e}. Utilisation de la question brute.")
            return [query], "Analyse directe de la question sans décomposition."



    def _deduplicate_docs(self, docs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dédoublonne les documents en conservant les usages distincts d'un même produit."""
        seen_ids = set()
        unique_docs = []
        for doc in docs_list:
            # Clé composite incluant le type d'usage pour éviter d'écraser des usages différents
            doc_id = doc.get("id") or f"{doc.get('numero_amm')}_{doc.get('culture')}_{doc.get('type_usage')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        return unique_docs



    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Formate le contexte en numérotant chaque source pour la traçabilité."""
        if not docs:
            return "Aucun document trouvé."

        formatted_chunks = []
        for idx, doc in enumerate(docs, start=1):
            formatted_chunks.append(
                f"--- [SOURCE {idx}] ---\n"
                f"Nom produit : {doc.get('nom_produit', 'Inconnu')}\n"
                f"AMM : {doc.get('numero_amm', 'N/A')}\n"
                f"Culture : {doc.get('culture', 'N/A')}\n"
                f"Type usage : {doc.get('type_usage', 'N/A')}\n"
                f"Contenu réglementaire :\n{doc.get('content', '')}\n"
            )
        return "\n".join(formatted_chunks)



    def run(self, query: str, top_k_per_subquery: int = 2) -> Dict[str, Any]:
        """Exécute le pipeline de raisonnement multi-étapes."""
        
        # Phase 1 : Décomposition
        sub_queries, thoughts_decomposition = self._decompose_query(query)
        
        # Phase 2 : Extraction multi-requêtes
        all_retrieved_docs = []
        for sub_q in sub_queries:
            docs = self.retriever.search(query=sub_q, top_k=top_k_per_subquery)
            all_retrieved_docs.extend(docs)

        # Phase 3 : Dédoublonnage & Formatage
        unique_docs = self._deduplicate_docs(all_retrieved_docs)
        context_str = self._format_context(unique_docs)
        
        user_prompt = f"CONTEXTE AGRÉGÉ :\n{context_str}\n\nQUESTION COMPLEXE :\n{query}"

        # Phase 4 : Synthèse par le LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )

            response_text = response.choices[0].message.content
        
        except Exception as e:
            response_text = f"[Erreur ReasoningAgent] Échec de la synthèse : {e}"

        return {
            "query": query,
            "thought_decomposition": thoughts_decomposition,
            "sub_queries": sub_queries,
            "response": response_text,
            "sources": unique_docs,
        }

    def close(self):
        if hasattr(self, "retriever") and self.retriever:
            self.retriever.close()