import json
import configparser
from pathlib import Path
from qdrant_client import QdrantClient
from groq import Groq

# 1. Chargement de la configuration
config = configparser.ConfigParser()
config_file = Path(".config")
if not config_file.exists():
    raise FileNotFoundError("Le fichier .config est manquant à la racine.")

config.read(config_file)

QDRANT_URL = config.get("qdrant", "QDRANT_URL")
QDRANT_API_KEY = config.get("qdrant", "QDRANT_API_KEY")
COLLECTION_NAME = config.get("qdrant", "COLLECTION_NAME", fallback="ephy_documents")

GROQ_API_KEY = config.get("groq", "GROQ_API_KEY")
GROQ_MODEL = config.get("groq", "GROQ_MODEL", fallback="llama-3.3-70b-versatile")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


def generate_benchmark_dataset(collection_name: str = COLLECTION_NAME, num_pairs: int = 20):
    """
    Récupère des produits distincts pour générer :
    - Des questions SIMPLES (monodoc, router_label=0)
    - Des questions COMPLEXES (comparaison/multidoc, router_label=1)
    - Des questions HORS_USAGE (rejet, router_label=0)
    """
    dataset = []

    # 1. Scroll large pour obtenir une diversité réelle de produits
    points, _ = qdrant.scroll(
        collection_name=collection_name,
        limit=250,
        with_payload=True,
        with_vectors=False
    )

    # 2. Dédoublonnage : 1 seul document par produit pour garantir que les paires sont différentes
    products_pool = {}
    for p in points:
        payload = p.payload or {}
        content = payload.get("content", "").strip()
        produit = payload.get("nom_produit", "").strip()
        amm = str(payload.get("numero_amm", "")).strip()

        if len(content) >= 40 and produit and amm:
            p_key = produit.upper()
            if p_key not in products_pool:
                products_pool[p_key] = {
                    "doc_id": str(payload.get("doc_id", p.id)),
                    "amm": amm,
                    "produit": produit,
                    "culture": payload.get("culture", ""),
                    "type_usage": payload.get("type_usage", ""),
                    "content": content
                }

    valid_docs = list(products_pool.values())
    print(f"[INFO] {len(valid_docs)} produits distincts identifiés dans Qdrant.")

    if len(valid_docs) < 2:
        raise ValueError("Nombre insuffisant de produits uniques dans la collection.")

    # 3. Assemblage par paires de produits distincts
    pair_count = 0
    for i in range(0, len(valid_docs) - 1, 2):
        doc_a = valid_docs[i]
        doc_b = valid_docs[i + 1]

        prompt = f"""Tu es un auditeur technique évaluant un système RAG phytosanitaire E-Phy et son routeur d'aiguillage.

Voici deux extraits officiels distincts :
--- DOCUMENT A ---
Produit : {doc_a['produit']} (AMM: {doc_a['amm']})
Culture : {doc_a['culture']} | Usage : {doc_a['type_usage']}
Texte : {doc_a['content']}

--- DOCUMENT B ---
Produit : {doc_b['produit']} (AMM: {doc_b['amm']})
Culture : {doc_b['culture']} | Usage : {doc_b['type_usage']}
Texte : {doc_b['content']}
---

Génère 3 cas de test au format JSON strict encapsulés dans une clé "tests" :

1. QUESTION SIMPLE (router_label = 0) :
- Porte UNIQUEMENT sur le Document A.
- Question factuelle directe (dose, DAR, culture, condition d'application).
- OBLIGATION de citer le nom du produit "{doc_a['produit']}".
- INTERDICTION de demander seulement le numéro d'AMM.

2. QUESTION COMPLEXE (router_label = 1) :
- Met en jeu les DEUX produits ("{doc_a['produit']}" ET "{doc_b['produit']}").
- Implique une comparaison (différence de dose, de culture autorisée, de restriction) OU un choix d'alternative entre les deux.

3. QUESTION HORS USAGE (router_label = 0) :
- Porte sur une maladie ou une culture totalement absente des deux documents pour ce produit.

Format attendu :
{{
  "tests": [
    {{
      "question": "...",
      "type": "simple",
      "router_label": 0,
      "reponse_attendue": "...",
      "expected_doc_ids": ["{doc_a['doc_id']}"],
      "expected_amm": ["{doc_a['amm']}"]
    }},
    {{
      "question": "...",
      "type": "complexe",
      "router_label": 1,
      "reponse_attendue": "...",
      "expected_doc_ids": ["{doc_a['doc_id']}", "{doc_b['doc_id']}"],
      "expected_amm": ["{doc_a['amm']}", "{doc_b['amm']}"]
    }},
    {{
      "question": "...",
      "type": "hors_usage",
      "router_label": 0,
      "reponse_attendue": "Information non mentionnée dans les documents officiels.",
      "expected_doc_ids": [],
      "expected_amm": []
    }}
  ]
}}
"""
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            items = data.get("tests", [])
            dataset.extend(items)
            pair_count += 1
            print(f"[OK] Paire {pair_count}/{num_pairs} : {doc_a['produit']} vs {doc_b['produit']} (+{len(items)} tests).")
        except Exception as e:
            print(f"[WARN] Erreur sur la paire ({doc_a['produit']}, {doc_b['produit']}): {e}")

        if pair_count >= num_pairs:
            break

    # 4. Ajout de cas de salutation / hors domaine (router_label = 2)
    dataset.extend([
        {
            "question": "Bonjour, comment fonctionne cet assistant ?",
            "type": "salutation",
            "router_label": 2,
            "reponse_attendue": "Message d'accueil et présentation des fonctionnalités.",
            "expected_doc_ids": [],
            "expected_amm": []
        },
        {
            "question": "Merci beaucoup pour votre aide, bonne journée !",
            "type": "salutation",
            "router_label": 2,
            "reponse_attendue": "Formule de politesse de clôture.",
            "expected_doc_ids": [],
            "expected_amm": []
        },
        {
            "question": "Quel temps fait-il aujourd'hui à Lyon ?",
            "type": "salutation",
            "router_label": 2,
            "reponse_attendue": "Hors domaine phytosanitaire.",
            "expected_doc_ids": [],
            "expected_amm": []
        }
    ])

    return dataset


if __name__ == "__main__":
    OUT_DIR = Path("tests/test_data")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH = OUT_DIR / "dataset.json"

    data = generate_benchmark_dataset(num_pairs=20)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCÈS] Dataset complet généré : {len(data)} exemples enregistrés dans '{OUT_PATH}'.")