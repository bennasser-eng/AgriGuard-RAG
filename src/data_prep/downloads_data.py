"""
Script to download and load the file of authorized product uses, clean it, and save it locally in CSV format.
"""

import io
import zipfile
import pandas as pd
import requests
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


# Download the data from the ZIP file ####
url_data = "https://www.data.gouv.fr/api/1/datasets/r/cb51408e-2b97-43a4-94e2-c0de5c3bf5b2"

print("Downloading the ZIP file...")
response = requests.get(url_data)

# Decompress the ZIP file directly in RAM
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    print("Files present in the archive :", z.namelist())

    # Load the CSV file of authorized product uses
    filename = "usages_des_produits_autorises_utf8.csv"

    with z.open(filename) as f:
        df_usages = pd.read_csv(f, sep=";", encoding="utf-8", low_memory=False)

    print(f"Total number of uses : {len(df_usages)}\n")

    print("List of columns ---")
    for col in df_usages.columns:
        print(f"- {col}")

    # Cleaning and filter of the data
    cols_a_garder = [
        'nom produit', 'numero AMM', 'Substances actives', 'identifiant usage', 'etat usage', 
        'dose retenue', 'dose retenue unite', 'delai avant recolte jour'
    ]
    df_sub = df_usages[cols_a_garder].copy()

    # Filter 1 : Keep only authorized uses
    df_sub = df_sub[df_sub['etat usage'].astype(str).str.lower() == 'autorisé']

    # Filter 2 : Filter on the Large Crops (e.g., Wheat, Corn, Rapeseed)
    mots_cles_gc = ['wheat', 'corn', 'rapeseed', 'oats', 'sunflower']
    pattern = '|'.join(mots_cles_gc)

    df_gc = df_sub[df_sub['identifiant usage'].astype(str).str.lower().str.contains(pattern, na=False)]

    print(f"Number of uses retained for the Large Crops : {len(df_gc)}")
    print(df_gc.head(3))

    # Local saving in CSV format in the working folder ######
    df_gc.to_csv(DATA_DIR / "ephy_grandes_cultures.csv", index=False, encoding="utf-8")
    print("File 'ephy_grandes_cultures.csv' saved locally !")


    # Load other useful files
    filename = "produits_condition_emploi_utf8.csv"
    with z.open(filename) as f:
        df_produits = pd.read_csv(f, sep=";", encoding="utf-8", low_memory=False)
    df_produits.to_csv(DATA_DIR / "ephy_produits_condition_emploi.csv", index=False, encoding="utf-8")
    print("File 'ephy_produits_condition_emploi.csv' saved locally !")