"""
Script to generate dataset for training the router model
The dataset consists of queries labeled as SIMPLE (0) or COMPLEX (1) or HORS_CONTEXT (2).
* HORS_CONTEXT queries are greetings or questions unrelated to the domain of agricultural regulations (Meteo, Finance ...).
* SIMPLE queries are straightforward, fact-based questions about products, AMM numbers, active substances, and other direct information.
* COMPLEX queries involve comparisons, negations, multiple constraints, or multi-step reasoning.
----------------
The generation process includes:
1. Defining templates for SIMPLE, COMPLEX, and HORS_CONTEXT queries.
2. Randomly filling in the templates with product names, AMM numbers, cultures, diseases, and substances or without for HORS_CONTEXT queries.
3. Shuffling the dataset to ensure a mix of query types.
4. Labeling each query with its corresponding type (SIMPLE, COMPLEX, or HORS_CONTEXT).
The dataset is designed to train a classifier that can distinguish between HORS_CONTEXT, SIMPLE and COMPLEX queries 
for routing purposes in the Agriguard system.
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


    # Hors_context queries (label = 2) : Greetings, unrelated questions (fiance, meteo, culture, etc.)
    templates_hors_context = [
        "Bonjour, comment ça va ?",
        "Salut, j'aime le {jeu}",
        "Qui a gagne entre Paris et Marseille {time} ?",
        "Quelle est la météo pour {time} ?",
        "Quel est le cours du marché pour le {produit} ?",
        "Quelle est la culture la plus rentable en {pays} ?",
        "Comment se déplace le sol en {pays} ?",
        "Est-ce que le sol se dégrade en {pays} ?",
        "Est-ce que la température est normale en {pays} ?",
        "Quel est le prix du pétrole {time} ?",
        "Comment faire pousser des {produit} en intérieur ?",
        "Quels sont les meilleurs investissements pour année {annee} ?",
        "Comment puis-je améliorer la qualité de mon sol ?",
    ]

    for _ in range(150):
        annee = random.choice(["2023", "2024", "2025", "2026"])     
        time = random.choice(["matin", "soir", "nuit", "hier", "aujourd'hui", "demain", "la semaine prochaine", "le 15 juin 2026"])
        pays = random.choice(["France", "Italie", "Espagne", "Allemagne", "Portugal", "USA", "Canada", "Australie"])
        produit = random.choice(["pétrole", "gaz", "bois", "charbon"])
        jeu = random.choice(["football", "basketball", "hockey", "handball"])
        template = random.choice(templates_hors_context)
        dataset.append({"query": template.format(annee=annee, time=time, pays=pays, produit=produit, jeu=jeu), "label": 2})


    random.shuffle(dataset)

    # Save dataset to JSON file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[OK] generated dataset : {len(dataset)} examples in '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    generate_router_data()