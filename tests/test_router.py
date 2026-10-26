"""
script de test pour le module IntentRouter.
"""

from src.router.schemas import QueryComplexity
from src.router.router import  IntentRouter

def run_router_tests():
    router = IntentRouter(model_path="models/router_model.joblib")   # Test mode heuristique/fallback

    test_queries = [
        ("Qui a gagne le dernier match de football ?", QueryComplexity.HORS_CONTEXT),
        ("Pouvez-vous me redonner les conditions d'application et la quantité par hectare pour le produit NEMO ?", QueryComplexity.SIMPLE),
        ("Bonjour, pouvez-vous m'aider avec un problème de septoriose ?", QueryComplexity.COMPLEXE),
        ("salut, j'aime le basket", QueryComplexity.HORS_CONTEXT),
        ("Quelle est la dose du produit NEMO ?", QueryComplexity.SIMPLE),
        ("Quel est le délai de rentrée du SARACEN DELTA ?", QueryComplexity.SIMPLE),
        ("Donne-moi un herbicide maïs sans nicosulfuron.", QueryComplexity.COMPLEX),
        ("Propose un programme contre la septoriose avec DAR < 30j et dose < 1.5 L/ha.", QueryComplexity.COMPLEX),
        ("Puis-je mélanger le CUTER et le NEMO ?", QueryComplexity.COMPLEX),
        ("Quel est le prix du pétrole aujourd'hui ?", QueryComplexity.HORS_CONTEXT),
        ("J'ai un problème de septoriose mais je dois récolter dans 3 semaines, il me faut un truc adapté.", QueryComplexity.COMPLEX),
        ("Quel est le prix du pétrole aujourd'hui ?", QueryComplexity.HORS_CONTEXT),

    ]

    print("=== TEST DU ROUTEUR D'INTENTIONS ===")
    for query, expected_complexity in test_queries:
        decision = router.route(query)
        status = "OK" if decision.complexity == expected_complexity else "ÉCHEC"
        print(f"\n[{status}] Query: '{query}'")
        print(f"       -> Décision: {decision.complexity.value} | Source: {decision.decision_source}")
        print(f"       -> Explication: {decision.reasoning}")


if __name__ == "__main__":
    run_router_tests()