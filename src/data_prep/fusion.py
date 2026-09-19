"""
Script to fuse the authorized product uses dataset with the product conditions dataset, clean it, 
and save it locally in json format.
"""

import pandas as pd
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


# Load our ephy_cultures dataset
df_gc = pd.read_csv(DATA_DIR / "ephy_grandes_cultures.csv")

# Load our ephy_conditions dataset
df_cond = pd.read_csv(DATA_DIR / "ephy_produits_condition_emploi.csv", sep=",")

# Filter of useful conditions (Délai de rentrée, EPI)
categories_utiles = ["Délai de rentrée", "Protection de l'opérateur"]
df_cond_filt = df_cond[df_cond["catégorie de condition d’emploi"].isin(categories_utiles)]

# Fuse with our Large Crops database on the AMM number
df_final = pd.merge(df_gc, 
    df_cond_filt[["numero AMM", "catégorie de condition d’emploi", "condition d’emploi libelle"]], 
    on="numero AMM", how="left"
    )

# Local saving of the combined CSV base
df_final.to_csv(DATA_DIR / "dataset_ephy_final.csv", index=False, encoding="utf-8")

# Transform to docs Markdown / JSON pour le RAG
documents = []
for idx, row in df_final.iterrows():
    dar = f"{row['delai avant recolte jour']} jours" if pd.notna(row['delai avant recolte jour']) else "Non renseigné / Non applicable"

    condition = row['condition d’emploi libelle'] if pd.notna(row['condition d’emploi libelle']) else "Aucune condition spécifique renseignée"

    cat_condition = row['catégorie de condition d’emploi'] if pd.notna(row['catégorie de condition d’emploi']) else "Général"

    doc_text = f"""# Produit : {row['nom produit']} (AMM : {row['numero AMM']})
        - **Substances actives :** {row['Substances actives']}
        - **Usage & Culture cible :** {row['identifiant usage']}
        - **Statut d'autorisation :** {row['etat usage']}
        - **Dose retenue :** {row['dose retenue']} {row['dose retenue unite']}
        - **Délai Avant Récolte (DAR) :** {dar}
        - **{cat_condition} :** {condition}
    """
    
    documents.append({
        "id": idx,
        "text": doc_text,
        "metadata": {
            "nom_produit": str(row['nom produit']),
            "amm": str(row['numero AMM']),
            "usage": str(row['identifiant usage']),
            "categorie_condition": str(cat_condition)
        }
    })

# Save as a JSON file
with open(DATA_DIR / "documents_rag_ephy.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"A combined dataset created ! {len(documents)} documents generated in 'documents_rag_ephy.json'.")