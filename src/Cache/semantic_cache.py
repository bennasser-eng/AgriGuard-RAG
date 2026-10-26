"""
Script to read the cache and identify if the query is matched with a known intent (greetings, out_of_scope, etc.)
"""

import json
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding


class SemanticCache:
    def __init__(self, cache_file: str = "data/semantic_cache.json", threshold: float = 0.85,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    
        self.cache_file = cache_file
        self.threshold = threshold
        self.model = TextEmbedding(model_name=model_name)

        # Chargement et alignement des listes
        self.queries = []
        self.responses = []
        self.intents = []
        self._load_cache()

        # Pré-encodage unique de la base du cache (Matrice N x 384)
        self.cache_embeddings = self._precompute_embeddings()



    def _load_cache(self) -> None:
        with open(self.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            intent = entry.get("intent", "HORS_CONTEXT")
            response = entry.get("response", "")
            for q in entry.get("queries", []):
                self.queries.append(q)
                self.responses.append(response)
                self.intents.append(intent)



    def _precompute_embeddings(self) -> np.ndarray:
        if not self.queries:
            return np.empty((0, 384))
        # Conversion du générateur FastEmbed en matrice NumPy
        return np.array(list(self.model.embed(self.queries)))



    def match(self, query: str) -> Optional[Tuple[str, str, float]]:
        """
        Check if the query matches a known intent in the cache.
        """
        if not query or not query.strip() or self.cache_embeddings.shape[0] == 0:
            return None

        # Encodage de la requête unique : format attendu (1, 384)
        query_embedding = np.array(list(self.model.embed([query])))

        # Calcul vectoriel direct (1, N)
        similarities = cosine_similarity(query_embedding, self.cache_embeddings)[0]

        # Récupération du meilleur candidat en une seule passe
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        print(f"Meilleur candidat : '{self.queries[best_idx]}' avec un score de {best_score:.4f}")

        if best_score >= self.threshold:
            return self.intents[best_idx], self.responses[best_idx], best_score

        return None


if __name__ == "__main__":
    # Test d'instanciation unique
    cache = SemanticCache()
    
    #test_query = "Quelle est la différence entre une orange et un cerveau ?"
    test_query = "Bonjour, comment ça va ?"
    result = cache.match(test_query)
    
    if result:
        intent, response, score = result
        print(f"[MATCH] Intent: {intent} | Score: {score:.4f}\nRéponse: {response}")
    else:
        print("[CACHE MISS] Requête à transmettre au routeur d'intention.")