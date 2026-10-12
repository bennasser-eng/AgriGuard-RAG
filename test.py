import os
from qdrant_client import QdrantClient

# Renseigne l'URL de ton cluster et ta clé API
URL = "https://c57d69e5-fcd7-47b7-8752-ea6ff241149b.australia-southeast1-0.gcp.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZjZmMDc5YTgtN2Y3Yi00Mjc1LTkxM2MtZGFhNDVkNTFiYmQ5In0.qYrwaSyl-iD1EO6eF4djLGWZrEfPqCzZQ-TJ87TUmhY"

client = QdrantClient(url=URL, api_key=API_KEY)

try:
    collections = client.get_collections()
    noms = [c.name for c in collections.collections]
    print(f"Collections trouvées : {noms}")

    if "ephy_documents" in noms:
        info = client.get_collection("ephy_documents")
        print(f"Statut : {info.status}")
        print(f"Nombre de points (documents) : {info.points_count}")
    else:
        print("La collection 'ephy_documents' n'existe pas sur ce cluster.")
except Exception as e:
    print(f"Erreur de connexion : {e}")
