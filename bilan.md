### **AgriGuard-RAG : Plateforme Agentique & RAG Hybride pour l'Audit Réglementaire Phytosanitaire en Temps Réel**.
dans les Grandes Cultures (blé, maïs, colza, tournesol, etc.).
Son rôle est de répondre de façon exacte et traçable aux questions sur la santé des plantes, le diagnostic des maladies, les traitements autorisés et les alternatives en biocontrôle.

### **Semaine1: La base des donnees:**
* Chargment des donnes ephy,
* Extraction des donnees utiles pour les grandes cultures, et les conditions d'utilisations,
* Creation du dataset au format json"

### **Semaine2: Routeur d'orchestre**
Le projet AgriGuard-RAG est une application RAG (Retrieval-Augmented Generation),
d'entreprise dédiée aux données réglementaires E-Phy (produits phytosanitaires).

### PHASE 2B (LE MOTEUR RAG HYBRIDE)
- Développer la couche de recherche pour les requêtes SIMPLES :
- Configuration de Qdrant (HNSW Dense + BM25 Sparse).
- Single-Stage Metadata Filtering (filtres stricts sur 'culture', 'AMM').
- Reranking par fusion RRF (Reciprocal Rank Fusion).

### Architecture choisie : Router-Driven Hybrid RAG (Déterministe d'abord, Agent ensuite).
- Étape 1 : Un Routeur d'Intentions ultra-rapide (< 1 ms) classe la requête.
- Étape 2 (Requêtes SIMPLES) : Aiguillage direct vers un RAG Hybride (BM25 + Dense Qdrant).
- Étape 3 (Requêtes COMPLEXES) : Aiguillage vers un Agent de Raisonnement (ReAct/LangGraph).
- Étape 4 : Validation finale par un Guardrail Réglementaire strict.,


### IMPLÉMENTATION DU ROUTEUR D'INTENTIONS (VALIDE & TESTÉ),
Le routeur décide si une requête est :
- QueryComplexity.SIMPLE  : Question factuelle directe (ex: dose, DAR, numéro AMM).
- QueryComplexity.COMPLEXE: Multi-contraintes, comparaison, négation (\"sans...\"), mélange.

### Fonctionnement du Routeur (Hybride) :**
1. Filtre Heuristique (Règles déterministes / Regex / Mots-clés logiques) :\n",
    "   Recherche les mots \"sans\", \"mélange\", \"programme\", ou les contraintes (ex: DAR < 30j).\n",
2. Classifieur Machine Learning (TF-IDF + LogisticRegression) :
Si l'heuristique ne tranche pas, le modèle Scikit-Learn (0.8 Mo) prédit la classe avec un score de confiance.

### COMMANDES DE REPRODUCTION (À LANCER DEPUIS LA RACINE)
* Exécution globale avec l'option -m (Recommandé)
1. Générer le dataset d'entraînement du routeur :
python -m src.router.generate_dataset

2. Entraîner le classifieur ML et sauvegarder dans models/ :
python -m src.router.train_router

3. Valider le routeur sur le jeu de tests réels :\n",
python -m tests.test_router


### PROCHAINE ÉTAPE : App et validation de la version 0 de ce projet
* repertoire app
* repertoire src/validation



# models de qroq:
qwen/qwen3.8-27b
openai/gpt-oss-safeguard-20b
whisper-large-v3-turbo
openai/gpt-oss-20b
meta-llama/llama-prompt-guard-2-86m
whisper-large-v3
canopylabs/orpheus-arabic-saudi
allam-2-7b
meta-llama/llama-prompt-guard-2-22m
openai/gpt-oss-120b
canopylabs/orpheus-v1-english

