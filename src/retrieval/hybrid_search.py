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


    def search(self,  query: str, top_k: int = 5, culture: Optional[str] = None, numero_amm: Optional[str] = None, type_usage: Optional[str] = None) -> List[Dict[str, Any]]:
        # Préfix the  requête with "query: "
        query_dense_text = f"query: {query}"
        dense_vec = list(self.dense_model.embed([query_dense_text]))[0]

        # For BM25, conserv the text whithout préfix e
        sparse_vec = list(self.sparse_model.embed([query]))[0]

        query_filter = self._build_filter(culture=culture, numero_amm=numero_amm, type_usage=type_usage)

        prefetch = [
            models.Prefetch(query=dense_vec.tolist(), using="dense", filter=query_filter, limit=top_k * 2),
            models.Prefetch(query=models.SparseVector(
                                              indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
                                              ), using="sparse", filter=query_filter, limit=top_k * 2,),
        ]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )

        retrieved_docs = []
        for point in results.points:
            retrieved_docs.append({
                "id": point.id,
                "score": point.score,
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
    try:
        test_query = "Quelle est la dose du produit NEMO ?"
        print(f"\n[TEST] Research for: '{test_query}'")
        
        hits = retriever.search(query=test_query, top_k=3)
        
        if not hits:
            print("[INFO] No result found.")
        else:
            for i, hit in enumerate(hits, 1):
                print(f"\n--- Result #{i} (RRF Score: {hit['score']:.4f}) ---")
                print(f"Product : {hit['nom_produit']} (AMM: {hit['numero_amm']})")
                print(f"Culture : {hit['culture']}")
                print(f"Type usage : {hit['type_usage']}")
                print(f"Content : {hit['content']}")
                print(f"Metadata : {hit['metadata']}")
    finally:
        retriever.close()