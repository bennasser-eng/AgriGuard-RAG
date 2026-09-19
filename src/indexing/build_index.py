"""
Script to build the Qdrant index for Ephy documents:
- The script loads the Ephy documents from a JSON file, generates dense and sparse embeddings using FastEmbed,
and inserts them into a local Qdrant collection.
- The collection is configured with dense and sparse vector embeddings, and metadata fields are indexed for filtering during search.
- This script is intended to be run once during the initial setup of the application.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.http import models
from .qdrant_setup import get_qdrant_client, setup_qdrant_collection, COLLECTION_NAME


# Files paths
DATA_FILE = Path("data/documents_rag_ephy.json")
BATCH_SIZE = 64     # Using batches to optimize CPU/RAM usage


def load_documents(file_path: Path) -> List[Dict[str, Any]]:
    """Load the json file: E-Phy."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"[OK] {len(docs)} documents loaded from '{file_path}'.")
    return docs


def build_and_push_index():
    """Generate embeddings and insert documents into the local Qdrant collection."""
    # 1. Load documents
    documents = load_documents(DATA_FILE)

    # 2. Initialize client and collection
    client = get_qdrant_client()
    setup_qdrant_collection(client, COLLECTION_NAME)

    # 3. Load the lightweight embedding models (FastEmbed CPU)
    print("[INIT] Loading embedding models (Dense multilingual-e5-small + Sparse BM25)...")
    dense_model  = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # 4. Process and insert in batches (Batches)
    total_docs = len(documents)
    print(f"[RUN] Beginning indexing of {total_docs} documents in Qdrant...")

    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Indexation Qdrant"):
        batch_docs = documents[i : i + BATCH_SIZE]
        
        # Extraction of the text content for embedding
        raw_texts = [doc.get("content", doc.get("text", "")) for doc in batch_docs]

        # Calcul of Dense vectors (Sémantique) and Sparse vectors (BM25)
        e5_texts = [f"passage: {text}" for text in raw_texts]
        
        dense_embeddings  = list(dense_model.embed(e5_texts))
        sparse_embeddings = list(sparse_model.embed(raw_texts))

        # Constrcution of the points to insert into Qdrant
        points = []
        for idx, (doc, text_content, dense_vec, sparse_vec) in enumerate(zip(batch_docs, raw_texts, dense_embeddings, sparse_embeddings)):
            point_id = i + idx
            
            metadata = doc.get("metadata", {})

            # Extraction by default of the metadata fields for filtering
            nom_prod = metadata.get("nom_produit", "")
            amm = metadata.get("amm", "")
            usage_raw = metadata.get("usage", "")

            # Parsing the usage field to extract culture and type_usage
            culture = usage_raw.split("*")[0].strip().upper() if "*" in usage_raw else usage_raw.upper()
            type_usage = usage_raw.split("*")[1].strip() if "*" in usage_raw else ""

            payload = {
                "doc_id": str(doc.get("id", point_id)),
                "numero_amm": str(amm).strip(),
                "nom_produit": str(nom_prod).strip(),
                "culture": culture,
                "type_usage": type_usage,
                "dar_jours": 0,
                "content": text_content,
                "metadata": metadata
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                    "dense": dense_vec.tolist(), 
                    "sparse": models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())
                    },
                    payload=payload
                )
            )

        # Insertion of the batch into Qdrant
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\n[OK] Indexing completed successfully ! {total_docs} documents are stored in 'data/qdrant_db/'.")


if __name__ == "__main__":
    build_and_push_index()