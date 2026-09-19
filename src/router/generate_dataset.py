"""
Script to generate dataset for training the router model
The dataset consists of queries labeled as SIMPLE (0) or COMPLEX (1).
* SIMPLE queries are straightforward, fact-based questions about products, AMM numbers, active substances, and other direct information.
* COMPLEX queries involve comparisons, negations, multiple constraints, or multi-step reasoning.
----------------
The generation process includes:
1. Defining templates for SIMPLE and COMPLEX queries.
2. Randomly filling in the templates with product names, AMM numbers, cultures, diseases, and substances.
3. Shuffling the dataset to ensure a mix of query types.
The dataset is designed to train a classifier that can distinguish between SIMPLE and COMPLEX queries 
for routing purposes in the AgriBot system.
"""

import json
import random
from pathlib import Path

OUTPUT_PATH = Path("data/router_training_data.json")

def generate_router_data():
    produits = ["NEMO", "SARACEN DELTA", "BELEM 0,8 MG", "KORIT 420 FS", "CUTER", "MESOSTAR", "AMISTAR"]
    amms = ["2080088", "2150432", "2010345", "2180991"]
    cultures = ["Blé", "Maïs", "Orge", "Tournesol", "Colza"]
    maladies = ["Désherbage", "Oïdium", "Septoriose", "Fusariose", "Rouille"]
    substances = ["nicosulfuron", "florasulame", "mésotrione", "pendiméthaline", "tébuconazole"]

    dataset = []

    # SIMPLE queries (label = 0) : Direct extraction, fact-based questions
    templates_simples = [
        "Quelle est la dose du produit {p} ?",
        "Quel est le numéro AMM de {p} ?",
        "Quelles sont les substances actives de {p} ?",
        "Quel est le délai de rentrée pour {p} ?",
        "Est-ce que {p} est autorisé sur {c} ?",
        "Donne-moi la fiche technique de l'AMM {a}.",
        "Quel est le DAR du produit {p} sur {c} ?",
        "Quels sont les EPI requis pour appliquer {p} ?",
        "Quelle est la quantité maximale de {p} par hectare ?"
    ]

    for _ in range(150):
        p = random.choice(produits)
        c = random.choice(cultures)
        a = random.choice(amms)
        t = random.choice(templates_simples)
        dataset.append({"query": t.format(p=p, c=c, a=a), "label": 0})

    # COMPLEX queries (label = 1) : Comparison, negation, multiple constraints, mixes
    templates_complexes = [
        "Trouve un herbicide sur {c} avec un DAR inférieur à {d} jours.",
        "Propose une alternative à {p} sans {s}.",
        "Quelle est la différence entre {p1} et {p2} sur {c} ?",
        "Puis-je mélanger {p1} et {p2} pour traiter {m} sur {c} ?",
        "Recommande un fongicide contre {m} sur {c} compatible avec un délai de rentrée de 6 heures.",
        "Propose un programme de pulvérisation alternant {s} et une autre molécule.",
        "Quel produit sur {c} permet de lutter contre {m} sans utiliser de {s} ?",
        "Propose un traitement pour {c} respectant une ZNT de {z}m et un DAR < {d}j."
    ]

    for _ in range(150):
        p1, p2 = random.sample(produits, 2)
        c = random.choice(cultures)
        m = random.choice(maladies)
        s = random.choice(substances)
        d = random.choice([24, 30, 48, 60])
        z = random.choice([5, 20, 50])
        t = random.choice(templates_complexes)
        dataset.append({"query": t.format(p=p1, p1=p1, p2=p2, c=c, m=m, s=s, d=d, z=z), "label": 1})

    random.shuffle(dataset)

    # Save dataset to JSON file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[OK] generated dataset : {len(dataset)} examples in '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    generate_router_data()