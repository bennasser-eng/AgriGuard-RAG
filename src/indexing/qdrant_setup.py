"""
Script to set up a local Qdrant collection for storing and indexing Ephy documents.
- The collection is configured with dense and sparse vector embeddings.
- Metadata fields are indexed for filtering during search.
This script is intended to be run once during the initial setup of the application.
"""

import warnings
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME = "ephy_documents"
DB_PATH = Path("data/qdrant_db")


def get_qdrant_client() -> QdrantClient:
    """Returns a Qdrant client connected to the local database path."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(DB_PATH))


def setup_qdrant_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """Sets up the Qdrant collection with dense and sparse vector configurations and metadata indexing."""
    collections = [col.name for col in client.get_collections().collections]
    if collection_name in collections:
        print(f"[INFO] The collection '{collection_name}' exist yet {DB_PATH}.")
        return

    print(f"[INIT] Creation of the collection '{collection_name}' localy...")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )

    print("[INIT] Configuration of filters for metadata...")
    
    fields_to_index = [
        ("numero_amm", models.PayloadSchemaType.KEYWORD),
        ("culture", models.PayloadSchemaType.KEYWORD),
        ("type_usage", models.PayloadSchemaType.KEYWORD),
        ("dar_jours", models.PayloadSchemaType.INTEGER)
    ]

    # Ignore the UserWarning specific to the local mode
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")
        for field_name, schema_type in fields_to_index:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    print(f"[OK] Collection '{collection_name}' initialized with success in '{DB_PATH}'.")


if __name__ == "__main__":
    qdrant_cli = get_qdrant_client()
    try:
        setup_qdrant_collection(qdrant_cli)
    finally:
        qdrant_cli.close()