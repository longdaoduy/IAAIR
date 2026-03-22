# 🔬 IAAIR — Intelligent Academic AI Retrieval

A hybrid AI-powered academic paper retrieval system that combines **vector search**, **knowledge graph traversal**, and **cross-modal visual search** to deliver precise, grounded, and verifiable answers to scientific queries.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Frontend UI](#frontend-ui)
- [Monitoring & Observability](#monitoring--observability)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Configuration](#configuration)

---

## Overview

IAAIR is a full-stack Retrieval-Augmented Generation (RAG) system purpose-built for academic literature search. It ingests papers from **OpenAlex** and **Semantic Scholar**, stores them across a **Neo4j** knowledge graph and **Milvus/Zilliz** vector database, and serves a unified FastAPI backend that powers multi-strategy hybrid search with AI-generated, citation-grounded answers.

### What makes IAAIR different?

| Capability | Description |
|---|---|
| **Hybrid Retrieval** | Graph-first, vector-first, or AI-auto-selected strategy per query |
| **Cross-Modal Search** | CLIP text→image search finds relevant figures and tables |
| **SciFact Verification** | Every AI answer is decomposed into atomic claims and verified against evidence |
| **Attribution Tracking** | AI response text is mapped back to source papers with confidence scores |
| **Conversation Memory** | Multi-turn sessions with context carryover for follow-up questions |
| **Input Guardrails** | Prompt injection and Cypher injection detection and sanitization |
| **Observability** | Prometheus metrics, Grafana dashboards, and per-operation latency tracking |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (index.html)                        │
│   Search · Visual Evidence · Graph Query · Performance · Models     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST API
┌──────────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Backend (main.py)                      │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Guardrails    │  │ Conversation │  │   ServiceFactory       │  │
│  │  sanitize_query│  │ Memory       │  │   (dependency inject)  │  │
│  └────────────────┘  └──────────────┘  └────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│               HybridRetrievalHandler (RAG Pipeline)                 │
│                                                                     │
│  1. Entity Extraction (AI + regex)                                  │
│  2. Template Selection (AI-ranked viable templates)                 │
│  3. Strategy Selection (graph_only / vector_first)                  │
│  4. Search Execution (Neo4j + Milvus + CLIP)                       │
│  5. Result Fusion (confidence scoring + visual re-ranking)          │
│  6. AI Response Generation (LLM synthesis with conversation ctx)   │
│  7. Attribution Building (map AI text → source papers)              │
│  8. SciFact Verification (claim extraction + label assignment)      │
└────┬──────────────┬──────────────┬──────────────┬──────────────┬────┘
     │              │              │              │              │
┌────▼────┐  ┌─────▼─────┐  ┌────▼────┐  ┌─────▼─────┐  ┌────▼────┐
│  Neo4j  │  │  Milvus/  │  │ SciBERT │  │   CLIP    │  │  LLM    │
│ (Graph) │  │  Zilliz   │  │ (embed) │  │ (visual)  │  │ (HF)   │
└─────────┘  └───────────┘  └─────────┘  └───────────┘  └─────────┘
```

---

## Key Features

### 🔍 Hybrid Search Pipeline

- **Graph-first**: Runs a Cypher query on Neo4j, then computes vector similarity for discovered papers. Best for queries with explicit entities (author names, paper IDs, venues, years).
- **Vector-first**: Discovers papers via SciBERT semantic search, then enriches with graph metadata. Best for pure topic/keyword queries.
- **Auto-selection**: An AI agent picks the optimal strategy based on extracted entities.

### 🧠 AI Response Generation

- Uses a local Hugging Face causal LM (configurable model) to synthesize answers grounded in retrieved evidence.
- Provides up to 8 papers with 800-character abstracts as context for richer RAG.
- Bracket-citation formatting (`[1]`, `[2]`) enables automatic attribution mapping.
- Conversation context from prior turns is injected for multi-turn follow-ups.

### 🔬 SciFact Verification

- AI responses are decomposed into atomic, verifiable claims.
- Each claim is labeled as **SUPPORTED**, **CONTRADICTED**, or **NO_EVIDENCE** against the retrieved papers.
- Verdict summary (Fully Supported / Partially Supported / Contradicted) is returned to the user.

### 📌 Attribution Tracking

- After AI response generation, the system matches response text to source papers using:
  - Exact and partial title matching
  - Author name matching
  - Bracket citation `[n]` matching
  - Abstract phrase overlap
- Each match produces an `AttributionSpan` with confidence score and character offsets.

### 💬 Conversation Memory

- Session-based multi-turn support with per-session history.
- Prior queries, AI responses, and discussed paper IDs are carried forward.
- LRU eviction with configurable timeout (default: 30 minutes) and max sessions (500).
- History is loadable in the frontend and clickable for query replay.

### 🛡️ Input Guardrails

- 12 prompt injection patterns detected and rejected (HTTP 400).
- 5 Cypher injection patterns detected and sanitized with `[FILTERED]`.
- Query length capped at 1000 characters with automatic truncation.
- Control character stripping.

### 🖼️ Cross-Modal Visual Search

- CLIP text→image embeddings search figures and tables collections in Milvus.
- SciBERT description embeddings provide text-level matches.
- Visual scores are combined with vector similarity for multi-modal re-ranking.
- Visual evidence (figures, tables) is displayed per-paper in the UI.

### ⚡ Performance & Caching

- **LRU caching** for query embeddings, search results, Cypher queries, and AI responses.
- **Persistent disk cache** for embeddings across restarts.
- **Performance monitor** tracks per-operation latency (vector search, graph search, AI response, fusion).
- **Prometheus metrics** exposed at `/metrics` for Grafana scraping.
- **Slow query detection** with configurable threshold.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Uvicorn, Pydantic |
| **LLM** | Hugging Face Transformers (configurable causal LM) |
| **Embeddings** | SciBERT (`allenai/scibert_scivocab_uncased`) |
| **Visual Embeddings** | CLIP (`openai/clip-vit-base-patch32`) |
| **Vector DB** | Milvus / Zilliz Cloud (`pymilvus`) |
| **Graph DB** | Neo4j (`neo4j` Python driver) |
| **Document Store** | MongoDB (optional, for raw paper storage) |
| **Metadata APIs** | OpenAlex, Semantic Scholar |
| **Monitoring** | Prometheus, Grafana, Grafana Alloy |
| **Frontend** | Single-file HTML/CSS/JS (served by FastAPI) |
| **GPU** | CUDA 12.x (optional, for LLM and embedding inference) |

---

## Project Structure

```
IAAIR/
├── main.py                          # FastAPI application (all endpoints)
├── requirements.txt                 # Python dependencies
├── start_api.sh                     # Startup script
│
├── clients/                         # External service clients
│   ├── huggingface/
│   │   ├── LLM_Client.py           # Causal LM (answer synthesis, claim extraction)
│   │   ├── SciBERTClient.py         # SciBERT text embeddings
│   │   └── CLIPClient.py            # CLIP cross-modal embeddings
│   ├── milvus/
│   │   └── MilvusClient.py          # Milvus/Zilliz vector search
│   ├── neo4j/
│   │   └── Neo4jClient.py           # Neo4j graph queries
│   ├── mongo/
│   │   └── MongoClient.py           # MongoDB document store
│   ├── metadata/
│   │   ├── OpenAlexClient.py        # OpenAlex API
│   │   └── SemanticScholarClient.py # Semantic Scholar API
│   └── prometheus/
│       ├── PrometheusClient.py      # Metrics exposition
│       └── PrometheusMetrics.py     # Metric definitions
│
├── models/
│   ├── configurators/               # Configuration classes
│   │   ├── LLMConfig.py             # Supported LLM models & params
│   │   ├── SciBERTConfig.py         # SciBERT config
│   │   ├── CLIPConfig.py            # CLIP config
│   │   ├── GraphDBConfig.py         # Neo4j connection config
│   │   ├── VectorDBConfig.py        # Milvus/Zilliz config
│   │   └── ...
│   ├── engines/                     # Core processing engines
│   │   ├── ServiceFactory.py        # Dependency injection container
│   │   ├── ResultFusion.py          # Score fusion + attribution building
│   │   ├── CacheManager.py          # Multi-layer LRU caching
│   │   ├── ConversationMemory.py    # Session-based multi-turn memory
│   │   ├── PerformanceMonitor.py    # Latency tracking & Prometheus
│   │   ├── ScientificReranker.py    # BGE reranker model
│   │   └── QueryClassifier.py       # Query type classification
│   ├── entities/                    # Pydantic data models
│   │   ├── retrievals/
│   │   │   ├── HybridSearchRequest.py
│   │   │   ├── HybridSearchResponse.py
│   │   │   ├── SearchResult.py
│   │   │   └── AttributionSpan.py
│   │   ├── ingestions/
│   │   └── evaluations/
│   └── schemas/                     # Additional schemas
│
├── pipelines/
│   ├── ingestions/                   # Data ingestion pipeline
│   │   ├── IngestionHandler.py      # OpenAlex → enrich → Neo4j + Milvus
│   │   ├── GraphNeo4jHandler.py     # Neo4j graph operations
│   │   ├── EmbeddingSciBERTHandler.py # Batch embedding generation
│   │   └── PDFProcessingHandler.py  # PDF → figures/tables extraction
│   ├── retrievals/
│   │   └── HybridRetrievalHandler.py # Main RAG pipeline (2500+ lines)
│   └── evaluations/
│       ├── MockDataEvaluator.py     # 50-question evaluation suite
│       ├── SciMMIRBenchmarkEvaluator.py # SciMMIR multi-modal benchmark
│       ├── SciMMIRDataLoader.py     # Dataset loading & caching
│       └── SciMMIRResultAnalyzer.py # Report generation
│
├── frontend/
│   └── index.html                   # Single-file frontend UI
│
├── data/
│   ├── graph_templates.json         # Neo4j Cypher query templates
│   ├── cypher_templates.json        # Additional Cypher templates
│   ├── evaluation_mock_data.json    # 50-question eval dataset
│   └── scimmir_cache/               # Cached SciMMIR benchmark data
│
├── monitoring/
│   ├── docker-compose.yml           # Prometheus + Grafana Alloy stack
│   ├── prometheus.yml               # Prometheus scrape config
│   ├── iaair_alerts.yml             # Alert rules
│   ├── alertmanager.yml             # Alert routing
│   ├── alloy-config.alloy           # Grafana Alloy config
│   └── grafana/                     # Grafana dashboard JSON
│
├── notebooks/                       # Data exploration notebooks
│   ├── openalex_data_import.ipynb
│   ├── semantic_scholar_data_import.ipynb
│   └── bioportal_*.ipynb
│
└── utils/
    ├── async_utils.py               # run_blocking helper (asyncio ↔ sync)
    └── metrics.py                   # Metric utilities
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Neo4j** instance (local or Aura)
- **Milvus** or **Zilliz Cloud** instance
- **CUDA 12.x** GPU (recommended for LLM inference; CPU fallback supported)
- **Docker** (optional, for monitoring stack)

### 1. Clone & Install

```bash
git clone https://github.com/longdaoduy/IAAIR.git
cd IAAIR

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root (or set environment variables):

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Milvus / Zilliz
MILVUS_HOST=localhost
MILVUS_PORT=19530
# Or for Zilliz Cloud:
# ZILLIZ_URI=https://your-cluster.zillizcloud.com
# ZILLIZ_TOKEN=your-api-key

# MongoDB (optional)
MONGO_URI=mongodb://localhost:27017

# Hugging Face (for gated models)
HF_TOKEN=hf_your_token

# OpenAlex / Semantic Scholar
OPENALEX_EMAIL=your-email@example.com

# Monitoring (optional)
GRAFANA_CLOUD_PROM_USERNAME=your-username
GRAFANA_CLOUD_API_KEY=your-api-key
GRAFANA_CLOUD_REMOTE_WRITE_URL=https://...
```

### 3. Start the API

```bash
# Option A: Direct
python main.py

# Option B: Using the startup script
chmod +x start_api.sh
./start_api.sh
```

The API starts at **http://localhost:8000**:
- **UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### 4. Ingest Papers (Optional)

Use the `/pull-papers` endpoint or the ingestion notebooks in `notebooks/` to populate Neo4j and Milvus with academic papers from OpenAlex.

---

## API Reference

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/hybrid-search` | Main hybrid search with AI response, verification, and attribution |
| `POST` | `/graph/query` | Execute raw Cypher queries against Neo4j |

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/pull-papers` | Pull papers from OpenAlex and process through the full pipeline |
| `GET` | `/download/{filename}` | Download generated JSON files |
| `GET` | `/ingestion/cursor-state` | Check ingestion cursor state |
| `DELETE` | `/ingestion/cursor-state` | Reset ingestion cursor |

### Graph Templates

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/graph/ai-templates` | List AI-selectable graph query templates |
| `GET` | `/graph/templates` | List user-saved custom templates |
| `POST` | `/graph/templates` | Save a custom Cypher template |
| `DELETE` | `/graph/templates/{name}` | Delete a saved template |

### Conversation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/conversation/history` | Get conversation turns for a session |
| `GET` | `/conversation/stats` | Get memory statistics (active sessions, total turns) |
| `POST` | `/conversation/clear` | Clear session or all conversation memory |

### Models

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/models/list` | List supported LLM models with specs |
| `POST` | `/models/switch` | Hot-swap the active LLM model |
| `GET` | `/models/stats` | LLM usage statistics by purpose |

### Evaluation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/evaluation/mock-data` | Run 50-question evaluation suite |
| `POST` | `/evaluation/scimmir-benchmark` | Run SciMMIR multi-modal benchmark |

### Performance & Cache

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/performance/stats` | Latency stats and bottleneck analysis |
| `GET` | `/performance/report` | Export detailed performance report |
| `GET` | `/cache/stats` | Cache hit rates and sizes |
| `POST` | `/cache/clear` | Clear all caches |
| `GET` | `/metrics` | Prometheus exposition format |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/ui` | Serve the frontend |
| `GET` | `/` | API info and endpoint listing |

---

## Frontend UI

The frontend is a single-file application served at `/ui` with the following panels:

| Panel | Description |
|---|---|
| **🔍 Search** | Hybrid search with strategy selector, graph template chips, AI toggle, conversation history, paper cards with attribution & visual evidence |
| **🕸️ Graph Query** | Direct Cypher query execution with template management |
| **📊 Performance** | Real-time latency stats, operation breakdown, slow queries |
| **💾 Cache** | Cache hit/miss rates, sizes, manual clear |
| **🤖 Models** | LLM model list, hot-swap, usage stats by purpose |
| **📡 Monitoring** | Links to Grafana dashboards |
| **🧪 Evaluation** | Run mock-data or SciMMIR benchmarks |

---

## Monitoring & Observability

### Prometheus + Grafana

```bash
cd monitoring
docker compose up -d
```

This starts:
- **Prometheus** (host networking, scrapes `localhost:8000/metrics`)
- **Grafana Alloy** (pushes metrics to Grafana Cloud)

### Tracked Metrics

- `iaair_search_duration_seconds` — search latency histogram
- `iaair_search_results_total` — result counts by source
- `iaair_cache_hits_total` / `iaair_cache_misses_total` — cache performance
- `iaair_llm_calls_total` — LLM invocations by purpose
- `iaair_verification_labels_total` — SciFact verification outcomes
- `iaair_template_used_total` — graph template usage frequency
- `iaair_search_strategy_total` — strategy selection distribution

### Alerting

Alert rules in `monitoring/iaair_alerts.yml` cover:
- High search latency (> 10s P95)
- High error rate
- Cache hit rate drops
- LLM error spikes

---

## Evaluation & Benchmarks

### Mock Data Evaluation

A 50-question evaluation suite covering diverse query types:

```bash
curl -X POST http://localhost:8000/evaluation/mock-data
```

Evaluates: retrieval quality (nDCG@k), AI response grounding, SciFact verification accuracy, latency budgets.

### SciMMIR Benchmark

Multi-modal scientific retrieval benchmark:

```bash
curl -X POST http://localhost:8000/evaluation/scimmir-benchmark
```

Evaluates: text-to-image retrieval, figure/table matching, cross-modal ranking quality. Compares against CLIP and SciBERT baselines.

---

## Configuration

### LLM Models

Models are defined in `models/configurators/LLMConfig.py`. Switch at runtime:

```bash
curl -X POST "http://localhost:8000/models/switch?model_name=meta-llama/Llama-3.2-3B-Instruct"
```

### Graph Templates

Query templates are stored in `data/graph_templates.json` and loaded at startup. Each template has:
- `cypher` — the parameterized Cypher query
- `description` — human-readable description
- `triggers` — entity keys required to activate (e.g., `paper_ids`, `author_names`)
- `param_builder` — method name for filling parameters

Templates can be reloaded without restart via the API.

### Cache Tuning

```bash
curl -X POST http://localhost:8000/performance/tune \
  -H "Content-Type: application/json" \
  -d '{"embedding_cache_size": 10000, "search_cache_ttl": 7200}'
```

---

## License

This project is developed as part of academic research. See the repository for license details.
