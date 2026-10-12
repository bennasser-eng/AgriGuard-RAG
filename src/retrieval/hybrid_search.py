"""
Script to perform a hybrid search with dense and BM25 embeddings.
The search is performed with a strict filter on metadata fields.
Using BM25 for the sparse vector embedding and sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for the dense embedding.
The search results are ranked using Reciprocal Rank Fusion (RRF) to combine the dense and sparse search results.
"""

from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.http import models

from src.indexing.qdrant_setup import get_qdrant_client, COLLECTION_NAME


class HybridRetriever:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = get_qdrant_client()

        # Models of embedding (FastEmbed on CPU)
        self.dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _build_filter(self, culture: Optional[str] = None,  numero_amm: Optional[str] = None, type_usage: Optional[str] = None):
        must_conditions = []

        if culture:
            must_conditions.append(models.FieldCondition(key="culture", match=models.MatchValue(value=culture.strip().upper())))
        if numero_amm:
            must_conditions.append(models.FieldCondition(key="numero_amm", match=models.MatchValue(value=str(numero_amm).strip())))
        if type_usage:
            must_conditions.append(models.FieldCondition(key="type_usage", match=models.MatchValue(value=type_usage.strip())))
        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)


    
    def _rrf_fusion(self, dense_results, sparse_results, top_k: int = 5, k: int = 60):
        """Fusion RRF manuelle compatible avec QdrantLocal."""
        scores = {}
        points_map = {}

        # 1. Rangs de la recherche dense
        for rank, point in enumerate(dense_results, start=1):
            points_map[point.id] = point
            scores[point.id] = scores.get(point.id, 0.0) + (1.0 / (k + rank))

        # 2. Rangs de la recherche sparse (BM25)
        for rank, point in enumerate(sparse_results, start=1):
            points_map[point.id] = point
            scores[point.id] = scores.get(point.id, 0.0) + (1.0 / (k + rank))

        # 3. Tri décroissant par score RRF
        sorted_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)[:top_k]
        
        return [(points_map[pid], scores[pid]) for pid in sorted_ids]

    def search(self, query: str, top_k: int = 5, culture: Optional[str] = None, numero_amm: Optional[str] = None, type_usage: Optional[str] = None) -> List[Dict[str, Any]]:
        # Encodage dense
        query_dense_text = f"query: {query}"
        dense_vec = list(self.dense_model.embed([query_dense_text]))[0]

        # Encodage sparse BM25
        sparse_vec = list(self.sparse_model.embed([query]))[0]

        query_filter = self._build_filter(culture=culture, numero_amm=numero_amm, type_usage=type_usage)

        # 1. Requête dense standard (compatible local)
        dense_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", dense_vec.tolist()),
            query_filter=query_filter,
            limit=top_k * 2
        )

        # 2. Requête sparse standard (compatible local)
        sparse_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name="sparse",
                vector=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            ),
            query_filter=query_filter,
            limit=top_k * 2
        )

        # 3. Fusion RRF côté application
        ranked_points = self._rrf_fusion(dense_hits, sparse_hits, top_k=top_k)

        # 4. Formatage des résultats
        retrieved_docs = []
        for point, rrf_score in ranked_points:
            retrieved_docs.append({
                "id": point.id,
                "score": rrf_score,
                "nom_produit": point.payload.get("nom_produit"),
                "numero_amm": point.payload.get("numero_amm"),
                "culture": point.payload.get("culture"),
                "type_usage": point.payload.get("type_usage"),
                "content": point.payload.get("content"),
                "metadata": point.payload.get("metadata", {})
            })

        return retrieved_docs



    def close(self):
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()



if __name__ == "__main__":
    retriever = HybridRetriever()