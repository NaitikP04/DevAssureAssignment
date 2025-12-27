# DevAssure RAG Chatbot

A file-based Multimodal RAG application for generating Test Cases from software documentation.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Loaders │→ │ Chunker  │→ │ Embedder │→ │ ChromaDB Store   │ │
│  │ PDF/DOC/ │  │ Smart    │  │ OpenAI   │  │ Vector + BM25    │ │
│  │ MD/TXT/  │  │ Splitting│  │ Embed    │  │ Hybrid Index     │ │
│  │ IMG      │  │          │  │          │  │                  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Hybrid Search│→ │  Reranking   │→ │   Safety Guards      │   │
│  │ BM25 + Vector│  │  FlashRank   │  │ - Relevance Filter   │   │
│  │ Ensemble     │  │  Top-N       │  │ - Injection Detect   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATION PIPELINE                          │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Multimodal LLM   │→ │  JSON Parse  │→ │   Evaluation     │   │
│  │ GPT-4 + Vision   │  │  Validation  │  │   Quality Check  │   │
│  │ Context Grounded │  │              │  │                  │   │
│  └──────────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               STRUCTURED TEST CASE OUTPUT (JSON)                 │
│  - Use Case Title, Goal, Preconditions                          │
│  - Test Data, Steps, Expected Results                           │
│  - Negative Cases, Boundary Conditions                          │
│  - Source References, Assumptions Made                          │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
DevAssureAssignment/
├── chatbot.py              # Main app - Gradio UI + RAG pipeline
├── ingest_database.py      # Ingests documents into ChromaDB
├── ingestion/              # Document loaders and chunking
├── guards/                 # Safety checks (relevance, injection detection)
├── evaluation/             # Output quality checks
├── utils/                  # Logging utilities
├── data/                   # Your input documents go here
├── chroma_db/              # Vector database (auto-generated)
└── logs/                   # Log files (auto-generated)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here
```

### 3. Add Your Documents
Put your files (PDF, DOCX, TXT, MD, PNG, JPG) in the `data/` folder.

### 4. Ingest Documents (Optional - for preprocessing)
```bash
python ingest_database.py
```
> **Note:** This step is optional. You can also upload files directly through the chatbot UI at runtime.

### 5. Run the Chatbot
```bash
python chatbot.py
```
Open `http://localhost:7860` in your browser.

## 📤 Runtime File Upload

You can upload files directly through the chatbot UI on the right sidebar:

1. **Select files** (PDF, TXT, MD, DOC/DOCX, PNG, JPG, JPEG)
2. **Click "Upload & Process"** - files are loaded, chunked, embedded, and stored
3. **Query immediately** - files are ready for retrieval

**Limits:** Max 5 files per upload, 10MB per file.

The **Database** section shows all indexed files. You can delete specific files or clear all runtime uploads.

## ⚙️ Advanced Settings (in the UI)

| Setting | What it does |
|---------|-------------|
| **Retrieval Top K** | How many documents to retrieve initially (default: 10). Higher = more context but slower. |
| **Rerank Top N** | After reranking, keep only the top N most relevant docs (default: 5). |
| **BM25 Weight** | Balance between keyword search (1.0) and vector search (0.0). Default 0.5 = equal mix. |
| **Enable Reranking** | When ON, uses a small ML model to re-score documents for better relevance. |

### When to adjust these:
- **Getting irrelevant results?** → Lower the BM25 weight (more vector search)
- **Missing obvious keyword matches?** → Raise the BM25 weight
- **Response too slow?** → Lower Top K and Top N
- **Missing context?** → Raise Top K

## 📝 Example Queries

- "Create test cases for flight filters"
- "Generate use cases for the dashboard feature"
- "Create negative test cases for user signup"

## 🛡️ Safety Features

1. **Relevance Filtering**: Low-relevance docs are filtered out
2. **Injection Detection**: Blocks "ignore previous instructions" type attacks
3. **Evidence Threshold**: Asks clarifying questions if context is insufficient
4. **Grounded Output**: Only uses info from retrieved documents

## 🧪 Run Evaluation Tests
```bash
python -m evaluation.evaluator
```

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-5.1
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Store**: ChromaDB
- **Retrieval**: LangChain (BM25 + Vector hybrid)
- **Reranking**: FlashRank
- **UI**: Gradio
- **Framework**: LangChain
- **IDE**: VSCode with Github Copilot
