"""
Script to set up and manage connections to the Qdrant vector database.
Supports Qdrant Cloud (via Streamlit Secrets or Environment Variables) 
with strict validation to prevent unindexed local fallback on cloud environments.
"""

import os
import warnings
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME = "ephy_documents"


def get_qdrant_client() -> QdrantClient:
    """Instancie un client Qdrant Cloud ou local avec détection stricte des identifiants."""
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # 1. Vérification dans Streamlit Secrets (en important le module de manière sécurisée)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            qdrant_url = st.secrets.get("QDRANT_URL")
            qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    except Exception:
        pass

    # 2. Vérification dans les variables d'environnement (exécution locale / CLI)
    if not qdrant_url:
        qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_api_key:
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # 3. Connexion à Qdrant Cloud si les paramètres sont définis
    if qdrant_url and qdrant_api_key:
        return QdrantClient(
            url=qdrant_url.strip(),
            api_key=qdrant_api_key.strip(),
            timeout=120.0
        )

    # 4. Détection d'environnement Streamlit Cloud : interdiction du fallback local
    is_streamlit_cloud = os.getenv("STREAMLIT_RUNTIME_ENV") is not None or os.path.exists("/mount/src")
    if is_streamlit_cloud:
        raise RuntimeError(
            "Identifiants Qdrant Cloud manquants dans Streamlit Secrets (QDRANT_URL ou QDRANT_API_KEY non définis). "
            "Le repli local est désactivé sur Streamlit Cloud pour éviter l'utilisation d'une base vide."
        )

    # 5. Fallback local pour développement offline uniquement
    return QdrantClient(path="./qdrant_data")


def setup_qdrant_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """Initialise la collection avec dense (384), sparse BM25 et index de métadonnées."""
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name in existing_collections:
        print(f"[INFO] La collection '{collection_name}' existe déjà.")
        return

    print(f"[INIT] Création de la collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )

    print("[INIT] Configuration des index de filtrage des métadonnées...")
    fields_to_index = [
        ("numero_amm", models.PayloadSchemaType.KEYWORD),
        ("culture", models.PayloadSchemaType.KEYWORD),
        ("type_usage", models.PayloadSchemaType.KEYWORD),
        ("dar_jours", models.PayloadSchemaType.INTEGER)
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")
        for field_name, schema_type in fields_to_index:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    print(f"[OK] Collection '{collection_name}' initialisée avec succès.")


if __name__ == "__main__":
    cli = get_qdrant_client()
    try:
        setup_qdrant_collection(cli)
    finally:
        cli.close()