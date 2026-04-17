# soulsborne.ai RAG Stack

soulsborne.ai is a local Retrieval-Augmented Generation (RAG) hobby project focused on Soulsborne game knowledge (Dark Souls and Bloodborne), built with:

- FastAPI backend for ingestion and chat orchestration
- Qdrant as the vector database
- vLLM serving a local OpenAI-compatible language model
- Open WebUI as the chat interface

The backend exposes OpenAI-style endpoints and injects retrieved context from Qdrant before forwarding prompts to vLLM.

## Architecture

1. Content ingestion
- Crawl pages from a sitemap (default: Bloodborne Fextralife sitemap)
- Extract clean text and metadata from HTML
- Chunk text, embed chunks, and upsert into Qdrant

2. Query flow
- Receive a chat completion request
- Embed user query and retrieve top chunks from Qdrant
- Re-rank chunks with a cross-encoder
- Build a grounded prompt (system prompt + retrieved context)
- Forward to vLLM and return OpenAI-style completion (streaming or non-streaming)

## Repository Layout

- `docker-compose.yml`: Orchestrates all services
- `backend/`: FastAPI app, crawler, ingestion pipeline, and query service
- `qdrant/`: Persisted Qdrant data directory
- `vllm/`: Persisted model cache for vLLM
- `open-webui/`: Open WebUI data and cache

## Services and Ports

- Open WebUI: `http://localhost:3000`
- Backend API: `http://localhost:8010`
- vLLM OpenAI-compatible API: `http://localhost:8000`
- Qdrant API: `http://localhost:6333`

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with compatible drivers (vLLM service is configured with `runtime: nvidia`)
- Hugging Face token with access to your target model

## Environment Variables

**IMPORTANT: Create a root `.env` file (same folder as `docker-compose.yml`) with at least:**

```env
# Required for vLLM model download
HF_TOKEN=your_hugging_face_token

# Model served by vLLM
LLM_NAME=microsoft/Phi-4-mini-instruct

# Shared API key used between backend and Open WebUI/vLLM
vLLM_API_KEY=change_me

# Backend config and tuning
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=soulsborne-collection
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_K=20
FINAL_K=5
```

## Quick Start

1. Start all services

```bash
docker compose up -d --build
```

2. Check containers

```bash
docker compose ps
```

3. Open the UI

- Navigate to `http://localhost:3000`
- Open WebUI is preconfigured to call the backend OpenAI-compatible endpoints

## Ingest Data into Qdrant

Trigger ingestion via backend endpoint:

```bash
curl -X POST http://localhost:8010/ingest \
	-H "Content-Type: application/json" \
	-d '{"sitemap_url":"https://bloodborne.wiki.fextralife.com/sitemap.xml","max_pages":100}'
```

Response:

```json
{
	"status": "started",
	"job_id": "<uuid>"
}
```

Notes:
- Ingestion runs in the background and can not be cancelled.
- Currently only supports sitemap based scraping
- Default sitemap is Bloodborne Fextralife when `sitemap_url` is omitted.
- Choose a positive `max_pages` value for predictable behavior.

## Chat API

### List available model

```bash
curl http://localhost:8010/v1/models
```

### Non-streaming completion

```bash
curl -X POST http://localhost:8010/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "soulsborne.ai",
		"stream": false,
		"messages": [
			{"role": "user", "content": "Who is Gehrman?"}
		]
	}'
```

### Streaming completion

```bash
curl -N -X POST http://localhost:8010/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "soulsborne.ai",
		"stream": true,
		"messages": [
			{"role": "user", "content": "Summarize the role of the Hunters Dream."}
		]
	}'
```

## Backend Highlights

- Query retrieval uses sentence-transformer embeddings and Qdrant vector search.
- Retrieved chunks are re-ranked with a cross-encoder before generation.
- System prompt enforces grounded answers and asks the model not to hallucinate beyond provided context.

## Troubleshooting

- vLLM fails to start:
	- Verify NVIDIA drivers and Docker GPU runtime support.
	- Confirm `HF_TOKEN` is valid and has model access.

- Empty or weak answers:
	- Run ingestion first.
	- Increase `max_pages` during ingestion.
	- Tune `RETRIEVAL_K` and `FINAL_K` in `.env`.

- Backend cannot reach Qdrant or vLLM:
	- Check service health with `docker compose logs backend qdrant vllm`.
	- Ensure all containers are on the same Compose network (default behavior).

## Development (Backend only)

From `backend/`:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

If running backend outside Docker, set environment variables so it can reach Qdrant and vLLM (for example `QDRANT_HOST=localhost` and `vLLM_URL=http://localhost:8000/v1/chat/completions`).
