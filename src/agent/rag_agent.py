"""
Script to implement the RAG Agent.
* The RAG Agent is responsible for handling user queries
* Call the Intent Router to determine the intent of the query
* Call the Reasoning Agent to generate a response based on the intent or the RAG Agent to retrieve relevant documents and generate a response.
* Then return the response to the user.
"""

from typing import List, Dict, Any, Optional
import ollama

from src.retrieval.hybrid_search import HybridRetriever
from src.router.router import IntentRouter
from src.agent.reasoning_agent import ReasoningAgent
from src.router.schemas import QueryComplexity

# Système Prompt to dell with hallucinations and ensure the agent strictly uses the provided documents for answering user queries.
SYSTEM_PROMPT = """Tu es un assistant expert de la réglementation phytosanitaire française (données E-Phy).
Ton rôle est de répondre aux questions des utilisateurs en t'appuyant STRICTEMENT sur les documents fournis en contexte.

Règles de conduite :
1. Si l'information exacte n'est pas présente dans les documents fournis, réponds explicitement : "Je ne dispose pas de cette information dans la base de données E-Phy."
2. Ne suppose jamais de valeurs et n'invente aucun délai, dosage ou numéro d'AMM.
3. Cite le nom du produit et son numéro d'AMM dans ta réponse lorsque c'est pertinent.
4. Reste direct, précis et factuel.
"""


class RAGAgent:
    def __init__(self, retriever=None, router=None, model_simple: str = "qwen2.5:3b", model_reasoning: str = "qwen2.5:3b"):
        self.retriever = retriever or HybridRetriever()
        self.router = router or IntentRouter()
        self.model_simple = model_simple
        self.reasoning_agent = model_reasoning
        self.reasoning_agent = ReasoningAgent(retriever=self.retriever, model_name= model_reasoning)

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Update the context with the relevant information from the retrieved documents."""
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


    def answer(self, query: str, top_k = 3, culture: Optional[str] = None, numero_amm = None, type_usage = None, use_llm: bool = True):
        """Exécute the complete pipeline : Routing -> Retrieval -> Formatting -> LLM Generation."""
        
        # 1. Routing d'intention
        intent_router_decision = self.router.route(query=query)
        is_complex = getattr(intent_router_decision, "complexity", None) == QueryComplexity.COMPLEX

        # Choice of the best pipeline
        if is_complex:
            print(f"[RAGAgent] Passage via ReasoningAgent ({self.reasoning_agent.model_name})...")
            result = self.reasoning_agent.run(query=query, top_k_per_subquery=top_k)
            result["query_complexity"] = intent_router_decision.complexity
            return result
        else:
            print(f"[RAGAgent] Aiguillage RAG Simple ({self.model_simple})...")
            # 2. Recherche hybride des documents
            retrieved_docs = self.retriever.search(query=query, top_k=top_k, culture=culture,numero_amm=numero_amm, type_usage=type_usage)

            # 3. Context's Formatting
            context_str = self._format_context(retrieved_docs)
            user_prompt = f"CONTEXTE:\n{context_str}\n\nQUESTION:\n{query}"

            # 4. Generation of the response with Ollama
            response_text = ""
            if use_llm:
                try:
                    response = ollama.chat(
                        model=self.model_simple,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        keep_alive=0  # Décharge le modèle immédiatement
                    )
                    response_text = response["message"]["content"]
                except Exception as e:
                    response_text = f"[Erreur Ollama] Impossible de générer la réponse : {e}"
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



if __name__ == "__main__":
    agent = RAGAgent()
    try:
         

        # TEST 2: Complex Query
        query2 = "Puis-je utiliser le produit NEMO sur blé et sur orge au même dosage ?"
        print(f"\n[TEST 2] Query : '{query2}'")
        result2 = agent.answer(query=query2)
        print(f"Complexité : {result2.get('query_complexity')}")
        
        if "sub_queries" in result2:
            print("\n**Sub Questions:**")
            for i, sq in enumerate(result2["sub_queries"], 1):
                print(f"{i}. {sq}")
                
        print("\n**Response generated by ReasoningAgent:**")
        print(result2["response"])

    finally:
        agent.close()