# AgriGuard-RAG
AgriGuard-RAG is a Retrieval-Augmented Generation (RAG) system designed to answer regulatory and technical queries regarding French agricultural phytosanitary products (E-Phy database).

By combining an ultra-low **latency intention router**, **hybrid vector retrieval**, and **automated reasoning pipelines**, AgriGuard ensures **deterministic compliance verification** while **handling complex multi-constraint queries**.


## Architecture Overview
The system follows a Router-Driven Architecture designed to optimize response latency, cost, and strict regulatory safety:

```text
[User Query] 
     │
     ▼
┌────────────────────────────────────────────────────────┐
│              Intent Router (< 1 ms Latency)            │
│  - Heuristic Engine (Multi-constraint Rules)   │
│  - ML Classifier (TF-IDF + Logistic Regression)        │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    [QueryComplexity.SIMPLE]        [QueryComplexity.COMPLEX]
            │                               │
            ▼                               ▼
 ┌──────────────────────┐        ┌──────────────────────┐
 │ Hybrid Search RAG    │        │ Reasoning Agent      │
 │ - Dense (BGE-Small)  │        │ - ReAct / LangGraph  │
 │ - Sparse (BM25)      │        │ - Multi-step Tools   │
 │ - Qdrant Single-Stage│        └──────────────────────┘
 │   Metadata Filtering │
 └──────────┬───────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│              Regulatory Guardrail & Response           │
└────────────────────────────────────────────────────────┘

```



## Features Implemented (Current State)

### 1. Data Pipeline & Structuring
* Processed raw agricultural records into 5,277 structured JSON chunks containing precise regulatory metadata (`numero_amm`, `culture`, `type_usage`, `dar_jours`).


### 2. Intent Router & Orchestrator
* Hybrid Fallback Mechanism:
1. Heuristic Filter: Immediately traps explicit patterns (exclusions like `"sans"`, tank mixes `"mélange"`, or numerical constraints like `DAR < 30j`).
2. Lightweight ML Classifier: Scikit-Learn pipeline (`TF-IDF + LogisticRegression`) classifying unstructured queries into `SIMPLE` or `COMPLEX` intents with sub-millisecond execution.


### 3. Local Hybrid Vector Storage (Qdrant)
* Embedded Vector Database: Embedded Qdrant instance running locally on disk (`data/qdrant_db/`) without external service requirements.
* Dual Embeddings:
* Dense: `transformer_sentences/`  for semantic understanding.
* Sparse: `Qdrant/bm25` for exact-match retrieval on regulatory IDs (AMM) and active substances.
* Single-Stage Filtering: Native metadata indexes for instant filtering on crop types (`culture`) and withdrawal periods (`dar_jours`).



### Installation
1. Clone the repository:
```bash
git clone https://github.com/bennasser-eng/AgriGuard-RAG.git
cd AgriGuard-RAG
```


2. Set up the virtual environment & install dependencies:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```



## Usage & Pipeline Execution

### Step 1: Train the Intent Router

Generate the training dataset and train the Scikit-Learn classification model:

```bash
# Generate training data
python -m src.router.generate_dataset

# Train & evaluate the model
python -m src.router.train_router

# Run integration tests for the router
python -m tests.test_router
```

### Step 2: Build the Vector Index

Initialize the local Qdrant database and index the 5,277 E-Phy regulatory documents using FastEmbed (CPU):

```bash
# Initialize Qdrant local schema
python -m src.indexing.qdrant_setup

# Populate vector store (Dense + BM25)
python -m src.indexing.build_index
```

### Step 3: Run Hybrid Search
Execute a test query using Reciprocal Rank Fusion (RRF):

```bash
python -m src.retrieval.hybrid_search
```



## Roadmap & Next Steps

* [x] Phase 1: Data Preparation & JSON Document Structuring
* [x] Phase 2.1: Low-Latency Intent Router & ML Classifier
* [x] Phase 2.2: Local Qdrant Indexing & Dual-Embedding Ingestion
* [ ] Phase 2.3: Hybrid Search Engine API Integration (`src/retrieval/`)
* [ ] Phase 3: Reasoning Agent for Complex Queries (LangGraph / ReAct)
* [ ] Phase 4: Regulatory Guardrails & Evaluation Framework (Ragas)
