
"""Script test the hybrid search."""

from src.retrieval.hybrid_search import HybridRetriever

def test_hybrid_search():
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
    

if __name__ == "__main__":
    test_hybrid_search()