# DocuChat: chat with your documents and get cited answers

**Upload your PDFs, Word files and Markdown, ask questions in plain language, and get answers grounded in your own documents, with a clickable citation for every claim.**

[![CI](https://github.com/gelevanog/docuchat-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/gelevanog/docuchat-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.140%2B-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL%2016-pgvector-4169E1?logo=postgresql&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

![DocuChat screenshot: an answer with citation chips and expanded source cards](docs/screenshot.png)

## Why DocuChat?

Teams keep their knowledge in handbooks, policies, contracts and product docs that nobody has time to search. A general chatbot will answer confidently but it can't see your files, and it may invent details. DocuChat answers **only from the documents you upload**. Every sentence points to the exact passage (file, page and section) it came from, so anyone can check the answer in one click. When the documents don't contain the answer, it says so and doesn't guess. DocuChat is self-hosted, so your documents stay on your own infrastructure, and it can use OpenAI, Anthropic Claude or, for demos and tests, a built-in offline model that needs no API key.

## Features

- **Multi-format ingestion.** PDF (with page numbers), DOCX (headings and tables), Markdown (heading hierarchy) and plain text.
- **Background processing.** Uploads return immediately. Parsing, chunking and embedding run as a background task, and the document moves through the statuses `pending → processing → ready / failed`.
- **Deduplication.** Files are hashed with SHA-256, so uploading identical content again does nothing. If a previous attempt failed, the re-upload retries it.
- **Token-aware chunking.** A recursive splitter works through paragraphs, then lines, then sentences, then words. Consecutive chunks overlap, and each chunk keeps its page and heading metadata.
- **Hybrid retrieval.** pgvector cosine similarity and PostgreSQL full-text search are fused with Reciprocal Rank Fusion. You can limit a question to selected documents.
- **Streamed, cited answers.** Tokens stream over Server-Sent Events. The model is told to cite sources as `[1]`, `[2]`, and the API returns only the sources the answer actually cites.
- **Conversations.** Chat history is stored in Postgres. Follow-up questions like *"and for part-timers?"* are rewritten into standalone search queries before retrieval.
- **Provider-agnostic LLM layer.** OpenAI (chat and embeddings), Anthropic Claude (chat) and a deterministic `fake` provider. You choose with environment variables and no code changes.
- **Retrieval evaluation.** `scripts/eval.py` reports hit-rate@k and MRR for vector-only, keyword-only and hybrid retrieval against a golden question set.
- **Production basics.** Alembic migrations, structured JSON logs, health check, OpenAPI docs, Docker Compose and a GitHub Actions CI pipeline.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (background task)"]
        U[Upload<br/>POST /api/documents] --> H{SHA-256<br/>seen before?}
        H -- yes --> N[Return existing<br/>document]
        H -- no --> S[(File storage)]
        S --> P[Parse<br/>pypdf / python-docx / md / txt]
        P --> C[Clean<br/>hyphenation, soft wraps]
        C --> K[Chunk<br/>recursive, token-aware,<br/>overlap + page/heading]
        K --> E[Embed in batches]
        E --> DB[(PostgreSQL<br/>pgvector HNSW +<br/>tsvector GIN)]
    end

    subgraph Query["Question answering"]
        Q[Question +<br/>chat history] --> R[Rewrite follow-up<br/>into standalone query]
        R --> V[Vector search<br/>cosine distance]
        R --> F[Full-text search<br/>ts_rank_cd]
        V --> RRF[Reciprocal Rank<br/>Fusion → top-k]
        F --> RRF
        RRF --> G[LLM with grounded prompt<br/>cite as 1, 2 or say I don't know]
        G --> A[SSE stream:<br/>tokens + citations]
    end

    DB -.-> V
    DB -.-> F
```

The sequence of a single chat request:

```mermaid
sequenceDiagram
    autonumber
    participant UI as Next.js UI
    participant API as FastAPI /api/chat
    participant DB as PostgreSQL + pgvector
    participant LLM as LLM provider

    UI->>API: POST /api/chat {message, conversation_id?, document_ids?}
    API->>DB: load / create conversation, recent history
    API->>DB: save user message
    API->>LLM: rewrite follow-up into standalone query (if history)
    API-->>UI: event: meta {conversation_id, rewritten_query}
    API->>LLM: embed query
    par hybrid retrieval
        API->>DB: vector search (HNSW, cosine)
    and
        API->>DB: full-text search (tsvector)
    end
    API->>API: Reciprocal Rank Fusion → top-k chunks
    API-->>UI: event: sources [numbered chunks]
    API->>LLM: system prompt + numbered sources + question (stream)
    loop streaming
        LLM-->>API: text delta
        API-->>UI: event: token {text}
    end
    API->>API: parse [n] citations
    API->>DB: save assistant message + cited sources
    API-->>UI: event: done {answer, citations}
```

## Quick start (Docker, no API keys)

Requirements: Docker with Compose v2.

```bash
git clone <this-repo-url> docuchat && cd docuchat
docker compose up --build
```

Then open **http://localhost:3000**, drag the files from [`sample_data/`](sample_data) into the sidebar and ask, for example, *"How many vacation days do full-time employees get?"*

- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health

By default the stack uses the offline **`fake` provider**. Embeddings are feature-hashed bags of words, and the "LLM" is extractive: it quotes the best-matching sentence from each retrieved source and cites it. This lets you check the whole pipeline (upload, ingestion, hybrid search, streaming, citations, history) without an account or any cost. Switch to a real model for fluent, synthesised answers.

You can also load the sample documents and run the evaluation from the command line:

```bash
for f in sample_data/*.md; do curl -s -F "file=@$f" http://localhost:8000/api/documents; echo; done
docker compose exec backend python scripts/eval.py   # also ingests sample_data/ if needed
```

> Ports 3000/8000 are taken? Set `FRONTEND_PORT`, `BACKEND_PORT`, `NEXT_PUBLIC_API_URL` and `CORS_ORIGINS` in `.env` (see [`.env.example`](.env.example)).

### Using real models

Copy `.env.example` to `.env`, choose the providers and add your keys, then rebuild:

```bash
cp .env.example .env
```

| Setup | `.env` |
|---|---|
| OpenAI for everything | `LLM_PROVIDER=openai`, `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY=sk-...` |
| Claude answers, OpenAI embeddings | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=sk-ant-...`, `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY=sk-...` |
| Claude answers, offline embeddings | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=sk-ant-...`, `EMBEDDING_PROVIDER=fake` |
| Self-hosted / OpenAI-compatible server (vLLM, Ollama, LiteLLM) | `LLM_PROVIDER=openai`, `OPENAI_BASE_URL=http://.../v1`, `OPENAI_CHAT_MODEL=...` |

```bash
docker compose up --build -d
```

The embedding size is fixed when the database is first migrated (`EMBEDDING_DIM`, default 1536, which matches `text-embedding-3-small`). If you switch to an embedding model with a different size, recreate the database (`docker compose down -v`) and re-upload your documents. Anthropic has no embeddings API, so pair Claude with OpenAI or the offline embedder.

### Local development without Docker

```bash
make dev-db                                    # pgvector on localhost:5432 (DEV_DB_PORT=... to change)
cp .env.example backend/.env                   # the backend reads backend/.env
make backend                                   # migrations + uvicorn --reload on :8000
make frontend                                  # Next.js dev server on :3000 (in another terminal)
```

## Configuration

All settings are environment variables. They are read by `backend/app/core/config.py` with pydantic-settings and validated at startup: for example, choosing `openai` without `OPENAI_API_KEY` fails fast.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `fake` | Chat model: `fake`, `openai` or `anthropic` |
| `EMBEDDING_PROVIDER` | `fake` | Embeddings: `fake` or `openai` |
| `EMBEDDING_DIM` | `1536` | Vector size stored in pgvector (fixed at first migration) |
| `OPENAI_API_KEY` | - | Required when an OpenAI provider is selected |
| `OPENAI_BASE_URL` | - | Optional OpenAI-compatible endpoint |
| `OPENAI_CHAT_MODEL` | `gpt-5-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ANTHROPIC_API_KEY` | - | Required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Claude model |
| `ANTHROPIC_MAX_TOKENS` | `16000` | Upper bound on answer length |
| `ANTHROPIC_EFFORT` | `medium` | Claude effort level: `low`, `medium` or `high` |
| `FAKE_STREAM_DELAY_MS` | `12` | Delay between streamed words of the fake provider |
| `CHUNK_SIZE_TOKENS` | `400` | Target chunk size (cl100k_base tokens) |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap between consecutive chunks |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding request |
| `MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `UPLOAD_DIR` | `./data/uploads` | Where original files are stored (`/data/uploads` volume in Docker) |
| `RETRIEVAL_TOP_K` | `6` | Chunks given to the model per question (overridable per request) |
| `RETRIEVAL_CANDIDATES` | `30` | Candidates taken from each retriever before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `HISTORY_TURNS` | `6` | Previous Q&A pairs used for rewriting and answering |
| `DATABASE_URL` | `postgresql+asyncpg://docuchat:docuchat@localhost:5432/docuchat` | Async SQLAlchemy URL (set automatically in Compose) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `docuchat` | Database credentials used by Compose |
| `DB_ECHO` | `false` | Log SQL statements |
| `ENVIRONMENT` | `development` | `development`, `production` or `test` |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FORMAT` | `console` | `console` (pretty) or `json` (default in the Docker image) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed browser origins |
| `BACKEND_PORT` / `FRONTEND_PORT` | `8000` / `3000` | Host ports published by Compose |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API URL used by the browser (inlined at frontend build time) |

## API

Interactive OpenAPI docs are available at `/docs` (Swagger UI) and `/redoc`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, database check and active providers |
| `POST` | `/api/documents` | Upload a file (multipart `file`). Returns `202` and `{document, duplicate}` |
| `GET` | `/api/documents` | List documents with their status |
| `GET` | `/api/documents/{id}` | One document (poll it until `status` is `ready`) |
| `DELETE` | `/api/documents/{id}` | Delete a document, its chunks and the stored file |
| `POST` | `/api/chat` | Ask a question. Streams SSE: `meta`, `sources`, `token`…, `done` (or `error`) |
| `GET` | `/api/conversations` | List conversations, most recent first |
| `GET` | `/api/conversations/{id}` | A conversation with its messages and citations |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation |

Example request, with the output shortened (real output from the offline provider):

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Are SMS codes accepted for MFA?", "top_k": 4}'
```

```text
event: meta
data: {"conversation_id":"01af6210-…","user_message_id":"c1c9b729-…","rewritten_query":"Are SMS codes accepted for MFA?"}

event: sources
data: {"sources":[{"id":1,"document_title":"Kestrel Grove IT Security Policy","page":null,"heading":"Accounts and Passwords","snippet":"Every person receives a personal account. …","score":0.032787, …}, …]}

event: token
data: {"text":"Here "}

…

event: done
data: {"message_id":"008d2483-…","answer":"Here is what the documents say:\n\n- SMS codes are not accepted as a second factor [1].","citations":[{"id":1,"chunk_id":"88841486-…","document_id":"ac5ec470-…","document_title":"Kestrel Grove IT Security Policy","filename":"it-security-policy.md","page":null,"heading":"Accounts and Passwords","snippet":"…","content":"…","score":0.032787}]}
```

Pass `"conversation_id"` to continue a conversation and `"document_ids": [...]` to search only those documents.

## Project structure

```text
docuchat/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # HTTP layer: documents, chat (SSE), conversations, health
│   │   ├── core/             # settings, structured logging, small text utilities
│   │   ├── db/               # SQLAlchemy 2 async models + session factory
│   │   ├── ingestion/        # parsers, cleaning, token-aware chunking, pipeline
│   │   ├── retrieval/        # hybrid search (pgvector + tsvector) and RRF
│   │   ├── llm/              # provider protocols, OpenAI / Anthropic / fake, prompts
│   │   ├── services/         # chat orchestration, documents, query rewriting
│   │   ├── container.py      # wires settings → providers → services
│   │   └── main.py           # FastAPI app factory
│   ├── migrations/           # Alembic (pgvector HNSW + GIN full-text indexes)
│   ├── scripts/eval.py       # retrieval evaluation: hit-rate@k, MRR
│   └── tests/                # unit + API/integration tests (pytest, no API keys)
├── frontend/src/
│   ├── app/                  # Next.js App Router entry
│   ├── components/           # sidebar, upload dropzone, chat, citation chips, source cards
│   ├── hooks/                # documents (with status polling), chat streaming, conversations
│   └── lib/                  # typed API client + incremental SSE parser
├── sample_data/              # original demo documents + eval/golden.yaml
├── docker-compose.yml        # db (pgvector) + backend + frontend
└── .github/workflows/ci.yml  # lint → test (with pgvector service) → build
```

## Key design decisions

**Hybrid search fused with Reciprocal Rank Fusion.** Dense vectors are good at paraphrase (*"holiday"* finds *"vacation"*). They are weak on exact tokens like product names, error codes, amounts or `CAMT.053`, and that's where keyword search shines. Running both and fusing them recovers the misses of each. RRF combines the two lists using **ranks only**. Cosine distances and `ts_rank` scores live on unrelated scales and can't be meaningfully added, so the rank-based approach needs no score normalisation or tuning. Both searches run inside PostgreSQL (HNSW and GIN indexes), so there is no separate vector database to operate. The [evaluation](#evaluation) below shows hybrid beating each retriever on its own.

**Recursive, token-aware chunking with overlap.** Chunks are measured in model tokens rather than characters, so they fit embedding and context limits predictably. The splitter prefers natural boundaries (paragraph, then line, then sentence, then word) and falls back to raw tokens only for pathological input. A 60-token overlap means a fact that straddles a boundary still appears whole in at least one chunk. Chunks never cross a page or section boundary, which keeps citations exact. Each chunk is embedded together with its document title and section heading, so a chunk that says "25 days" also carries the context that it's about vacation.

**Citations are part of the contract, not decoration.** The prompt numbers every source and requires `[n]` citations, or an explicit "I don't know" when the sources don't cover the question. The server parses the citations and returns only the sources actually used, so the UI can render verifiable source cards. Source text is escaped and marked as data, which makes it harder for instructions hidden inside a document to take over the prompt. This makes answers auditable and exposes hallucinations instead of hiding them.

**A small provider interface.** The application depends on two protocols, `ChatModel` (`stream`, `complete`) and `EmbeddingModel` (`embed`), and never on a vendor SDK directly. Adding a provider (Azure OpenAI, Bedrock, a local model) is one adapter class. The deterministic `fake` provider is what makes the test suite, CI and the demo run with no keys, no network and no cost.

**Follow-up rewriting before retrieval.** *"And for adoptive parents?"* retrieves nothing useful on its own. With a real LLM the question is rewritten into a standalone query using the chat history. The offline provider uses a heuristic instead: it anchors short or "and/what about…" questions to the previous question.

**Boring, reliable plumbing.** Uploads are deduplicated by content hash with a unique constraint, which also handles concurrent duplicates. Ingestion is idempotent: a retry replaces any partial chunks. Failures are recorded on the document instead of being lost in logs. Migrations run automatically on container start.

## Evaluation

`backend/scripts/eval.py` ingests `sample_data/` (a no-op if it's already there) and runs 18 golden questions from [`sample_data/eval/golden.yaml`](sample_data/eval/golden.yaml). Each question is sent through vector-only, keyword-only and hybrid retrieval. It is scored at two levels:

- **document**: any chunk from the expected document counts as a hit;
- **chunk**: the chunk must also contain the answering passage (`expected_text`).

Real output with the offline `fake` embedder (`docker compose exec backend python scripts/eval.py`):

```text
Embedding provider: fake:hashing-1536 | questions: 18 | depth: 5

Per-question rank of the first relevant chunk (hybrid retrieval)

question                                                                   doc  chunk
How many vacation days do full-time employees get per year?                  1      1
Can I carry unused holiday over to next year?                                1      1
When do I need a doctor's note if I'm sick?                                  1      1
How long is parental leave for the non-birthing parent?                      1      1
What is the maximum hotel cost I can expense in London?                      1      1
How big is the annual learning budget?                                       1      1
How much does the Team plan cost?                                            1      1
Do I need a credit card for the free trial?                                  1      1
In which country is customer data stored?                                    1      1
What is the refund policy for annual subscriptions?                          1      1
What are the support hours and response times?                               1      1
Which file formats can bank statements be imported in?                       1      1
What is the minimum password length?                                         1      1
Are SMS codes allowed for two-factor authentication?                         2      2
What should I do if my laptop is stolen?                                     1      1
Can I paste customer data into ChatGPT or other AI tools?                    1      1
How quickly must the regulator be notified about a personal data breach?     1      2
What happens if I click the link in a phishing test?                         1      1

mode/level            hit@1    hit@3    hit@5      mrr
vector/document       0.833    1.000    1.000    0.917
vector/chunk          0.833    1.000    1.000    0.917
keyword/document      0.889    1.000    1.000    0.944
keyword/chunk         0.833    1.000    1.000    0.917
hybrid/document       0.944    1.000    1.000    0.972
hybrid/chunk          0.889    1.000    1.000    0.944
```

The dataset is intentionally tiny (3 documents), so treat these numbers as a smoke test and a template, not a benchmark. The useful part is the harness. Point `--dataset` and `--docs` at a client's real documents and questions, then compare chunk sizes, embedding models or `RRF_K` on facts instead of impressions. `--json` prints machine-readable results for CI.

## Testing

```bash
make test-db     # throwaway pgvector container on localhost:55433
make test        # 81 tests: unit + API/integration, all with the fake provider (no keys)
make lint        # ruff, ruff format --check, mypy --strict, eslint, tsc
```

- **Unit tests** cover the chunker (budget, overlap, boundary preference, hard splits), RRF math, prompt building and escaping, citation parsing, parsers (generated PDF and DOCX fixtures), text cleaning, query rewriting, the fake provider, and the OpenAI and Anthropic adapters against stub clients.
- **Integration tests** run the real FastAPI app against PostgreSQL + pgvector with Alembic migrations applied: upload → background ingestion → status, deduplication, failed ingestion, SSE chat with citations, follow-up rewriting, document filters and conversation persistence.
- If the database is unreachable, the integration tests are **skipped** and the unit tests still run. CI sets `REQUIRE_TEST_DB=1`, so there a missing database is a failure.

## Roadmap

- OCR for scanned PDFs (e.g. Tesseract or a document-AI service) and HTML/Confluence connectors
- A cross-encoder re-ranking step after fusion, measured with the eval harness
- Authentication and per-workspace document permissions
- A durable job queue (e.g. arq/Celery) for large batch ingestion, with retries and progress
- Answer-quality evaluation (faithfulness and citation precision) with an LLM judge

## License

[MIT](LICENSE) © 2026 Ivan Savchenko
