"""
script of our benchmark for the hybrid search and router
"""

import json, os, configparser
import time
from pathlib import Path
from typing import List, Dict, Any

config_path = Path(".config")
if config_path.exists():
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if cfg.has_section("qdrant"):
        os.environ["QDRANT_URL"] = cfg.get("qdrant", "QDRANT_URL")
        os.environ["QDRANT_API_KEY"] = cfg.get("qdrant", "QDRANT_API_KEY")
        os.environ["COLLECTION_NAME"] = cfg.get("qdrant", "COLLECTION_NAME")
    
from src.retrieval.hybrid_search import HybridRetriever
from src.router.router import  IntentRouter



def run_benchmark(dataset_path: str = "tests/test_data/dataset.json", k: int=5):
    path = Path(dataset_path)
    retriever = HybridRetriever()
    if not path.exists():
        raise FileNotFoundError(f"Fichier de test introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 1. MÉTRIQUES DU ROUTEUR
    router_correct = 0
    router_total = 0
    router_by_class = {0: {"correct": 0, "total": 0},
                       1: {"correct": 0, "total": 0},
                       2: {"correct": 0, "total": 0}}

    # 2. MÉTRIQUES DU RETRIEVAL
    retrieval_queries_count = 0
    hit_counts = 0
    recalls = []
    precisions = []
    mrr_scores = []
    latencies_ms = []

    print(f"Lancement du benchmark sur {len(data)} requêtes (Top-k = {k})...\n")


    # Mapping vers les labels entiers du dataset
    mapping = {
        "SIMPLE": 0,
        "COMPLEXE": 1,
        "COMPLEX": 1,
        "SALUTATION": 2,
        "CHAT": 2
    }

    for item in data:
        query = item["question"]
        expected_route = item.get("router_label")
        expected_amms = [str(a).strip() for a in item.get("expected_amm", []) if a]
        q_type = item.get("type", "inconnu")

        #  Évaluation du Routeur
        if expected_route is not None:
            decision = IntentRouter().route(query=query)
            val = getattr(decision.complexity, "value", str(decision.complexity)).upper()

            predicted_route = mapping.get(val, -1)

            router_total += 1
            if predicted_route == expected_route:
                router_correct += 1
                if expected_route in router_by_class:
                    router_by_class[expected_route]["correct"] += 1
            if expected_route in router_by_class:
                router_by_class[expected_route]["total"] += 1

        # Évaluation du Retrieval
        if expected_amms and q_type != "salutation":
            retrieval_queries_count += 1
            
            t0 = time.perf_counter()
            results = retriever.search(query=query, top_k=k)
            lat_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(lat_ms)

            retrieved_amms = [str(r.get("numero_amm", "")).strip() for r in results]

            # Matching basé sur les AMMs attendues
            matched = [amm for amm in retrieved_amms if amm in expected_amms]
            unique_matched = set(matched)

            # Hit Rate@k (au moins 1 document pertinent trouvé)
            has_hit = 1 if len(unique_matched) > 0 else 0
            hit_counts += has_hit

            # Recall@k : proportion des AMM attendues retrouvées
            recall = len(unique_matched) / len(set(expected_amms))
            recalls.append(recall)

            # Precision@k : proportion de documents retournés qui sont pertinents
            precision = len(matched) / k
            precisions.append(precision)

            # Reciprocal Rank (MRR) : 1 / rang du premier match pertinent
            rr = 0.0
            for rank, amm in enumerate(retrieved_amms, start=1):
                if amm in expected_amms:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)


    # ---------
    # AFFICHAGE DES RÉSULTATS
    print("=" * 50)
    print(" 1. PERFORMANCE DU ROUTEUR")
    print("=" * 50)
    if router_total > 0:
        global_acc = (router_correct / router_total) * 100
        print(f"Accuracy Globale        : {global_acc:.2f}% ({router_correct}/{router_total})")
        for cls_id, label_name in [(0, "SIMPLE"), (1, "COMPLEXE"), (2, "SALUTATION")]:
            stats = router_by_class[cls_id]
            if stats["total"] > 0:
                acc_cls = (stats["correct"] / stats["total"]) * 100
                print(f" - Précision [{label_name:<10}] : {acc_cls:.1f}% ({stats['correct']}/{stats['total']})")
    else:
        print("Aucun label de routage présent dans le dataset.")

    print("\n" + "=" * 50)
    print(f" 2. PERFORMANCE DU RETRIEVAL QDRANT (k={k})")
    print("=" * 50)
    if retrieval_queries_count > 0:
        mean_hit_rate = (hit_counts / retrieval_queries_count) * 100
        mean_recall = (sum(recalls) / len(recalls)) * 100
        mean_precision = (sum(precisions) / len(precisions)) * 100
        mean_mrr = sum(mrr_scores) / len(mrr_scores)
        mean_lat = sum(latencies_ms) / len(latencies_ms)

        print(f"Hit Rate@{k}             : {mean_hit_rate:.2f}%")
        print(f"Recall@{k}               : {mean_recall:.2f}%")
        print(f"Precision@{k}            : {mean_precision:.2f}%")
        print(f"MRR@{k}                  : {mean_mrr:.3f}")
        print(f"Latence moyenne         : {mean_lat:.1f} ms")
    else:
        print("Aucune requête évaluable pour le retrieval.")
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark(dataset_path="tests/test_data/dataset.json", k=10)