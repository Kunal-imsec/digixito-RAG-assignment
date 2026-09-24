# Digixito RAG Assignment

A production-oriented **PDF Document Q&A system** built with FastAPI, ChromaDB, Sentence Transformers, and Groq. The system supports document ingestion, semantic retrieval, grounded answers, citations, streaming responses, persistent conversation history, and request-level observability.

## Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │ Upload / Ask / SSE   │
                    └──────────┬──────────┘
                               │ HTTP
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │ /upload /ask /health│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │ PDF Process │  │ Conversation │  │ Observability│
       │ Extract     │  │ JSON History │  │ Logs/Metrics │
       │ Clean       │  └─────────────┘  └─────────────┘
       │ Chunk       │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Embeddings  │
       │ MiniLM-L6   │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │  ChromaDB   │
       │ Persistent   │
       │ Vector Store │
       └──────┬──────┘
              │ Top-K
              ▼
       ┌─────────────┐
       │ Grounded    │
       │ Prompt      │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Groq        │
       │ GPT-OSS 120B│
       └──────┬──────┘
              │ SSE
              ▼
       ┌─────────────┐
       │ Answer +    │
       │ Citations   │
       └─────────────┘
```

## Tech Stack

* **Backend:** FastAPI, Python
* **LLM:** Groq API using `openai/gpt-oss-120b`
* **Embeddings:** Sentence Transformers `all-MiniLM-L6-v2`
* **Vector DB:** ChromaDB with persistent storage
* **PDF Processing:** PyMuPDF
* **Frontend:** Streamlit
* **Streaming:** Server-Sent Events (SSE)
* **Conversation Memory:** JSON-based persistent history
* **Observability:** structured logging + request IDs + latency metrics

## RAG Pipeline

```text
PDF
 ↓
PyMuPDF extraction
 ↓
Text cleaning
 ↓
500-character chunks
 ↓
50-character overlap
 ↓
all-MiniLM-L6-v2 embeddings
 ↓
Persistent ChromaDB
 ↓
Semantic Top-K retrieval
 ↓
Grounded prompt + conversation history
 ↓
Groq GPT-OSS 120B
 ↓
Streaming answer + source citations
```

### Chunking Strategy

* Chunk size: **500 characters**
* Overlap: **50 characters**
* Processing is performed per PDF page.
* Each chunk stores:

  * filename
  * page number
  * chunk ID
  * document ID
  * conversation ID

### Retrieval Strategy

The current implementation uses **dense semantic retrieval**:

1. Embed the user query using `all-MiniLM-L6-v2`.
2. Query ChromaDB using cosine similarity.
3. Restrict results to the current `conversation_id`.
4. Return the configured **Top-K = 5** chunks.
5. Pass retrieved context to the LLM.

No BM25 or hybrid retrieval is currently used.

### LLM Strategy

The system uses:

* Provider: **Groq**
* Model: **`openai/gpt-oss-120b`**
* Temperature: **0.1**
* Maximum output: **2048 tokens**
* Streaming enabled through Groq's OpenAI-compatible API.

The prompt instructs the model to answer using retrieved document context and return **`insufficient information`** when the available context does not support the answer.

## Features

* PDF upload and validation
* PDF text extraction and cleaning
* Configurable chunking
* Local embedding generation
* Persistent ChromaDB storage
* Conversation-scoped retrieval
* Grounded document Q&A
* Source/page citations
* Streaming responses
* Persistent conversation history
* Request IDs
* Latency and request metrics
* Health endpoint
* Streamlit test/demo UI
* Graceful handling of malformed and unsupported files

## Setup

### 1. Clone

```bash
git clone https://github.com/Kunal-imsec/digixito-RAG-assignment.git
cd digixito-RAG-assignment
```

### 2. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env
```

Set:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Optional configuration:

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=data/chromadb
CHROMA_COLLECTION_NAME=documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
CONVERSATION_DIR=data/conversations
MAX_CONVERSATION_HISTORY=10
LOG_LEVEL=INFO
UPLOAD_DIR=data/uploads
```

### 5. Start backend

```bash
uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Health:

```text
http://127.0.0.1:8000/health
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start Streamlit UI

In another terminal:

```bash
pip install -r requirements-frontend.txt
streamlit run streamlit_app.py
```

## API

### Create conversation

```http
POST /conversations
```

Response:

```json
{
  "conversation_id": "..."
}
```

### Upload PDF

```http
POST /upload
Content-Type: multipart/form-data
```

Parameters:

```text
conversation_id=<id>
file=<pdf>
```

Response includes:

```json
{
  "message": "PDF uploaded and processed successfully.",
  "file_name": "document.pdf",
  "num_pages": 5,
  "num_chunks": 20,
  "conversation_id": "..."
}
```

### Ask a question

```http
POST /ask
Content-Type: application/json
```

Request:

```json
{
  "question": "What technology stack does the engineering team use?",
  "conversation_id": "..."
}
```

The endpoint returns an **SSE stream** containing:

* `token` events for incremental generation
* `done` event containing the final answer and citations
* `error` event when generation fails

### Health

```http
GET /health
```

Returns application status, configured models, ChromaDB status, and runtime metrics.

## Evaluation

The repository contains a **20-case RAG pass/fail evaluation** covering:

* factual document questions
* retrieval
* grounded responses
* insufficient-information handling
* follow-up questions
* technology-stack questions
* numeric facts
* file validation
* malformed PDFs

Recorded evaluation result:

**20 / 20 test cases passed**

The evaluation includes both positive RAG cases and negative/file-validation cases.

Test artifacts:

```text
rag_test_passorfail.csv
evaluation/evaluation.csv
TEST_CHECKLIST.md
rag_test_document.pdf
```

## Limitations

* PDF processing currently extracts text only; scanned/image-only PDFs require OCR and are not supported.
* Retrieval is dense semantic search only; there is no BM25/hybrid reranking.
* Chunking is character-based rather than token/semantic based.
* ChromaDB and conversation history use local persistent storage.
* The system requires a Groq API key for generation.
* Conversation isolation depends on the supplied `conversation_id`.
* No authentication or multi-user authorization layer is included.
* The Streamlit UI is intended as a local test/demo frontend.

## Project Structure

```text
.
├── app/
│   ├── api/             # FastAPI routes
│   ├── conversation/    # Persistent conversation history
│   ├── embeddings/      # Sentence Transformer embeddings
│   ├── ingestion/       # PDF extraction, cleaning, chunking
│   ├── llm/             # Groq client and prompts
│   ├── observability/   # Logging and metrics
│   ├── retrieval/       # Semantic retrieval
│   ├── vectorstore/     # ChromaDB integration
│   ├── config.py        # Environment configuration
│   └── main.py          # FastAPI application
├── data/
│   ├── chromadb/        # Persistent vector store
│   ├── conversations/   # Conversation history
│   └── uploads/         # Uploaded PDFs
├── evaluation/          # Evaluation data
├── streamlit_app.py     # Test/demo UI
├── requirements.txt
├── requirements-frontend.txt
└── .env.example
```

## License

This project was developed as a technical assignment/prototype and is provided for evaluation and demonstration purposes.
