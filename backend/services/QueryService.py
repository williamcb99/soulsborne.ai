import os
from typing import Any, Dict
import httpx
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from classes.ChatRequest import ChatRequest
import asyncio
import logging


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "soulsborne-collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MODEL_NAME = os.getenv("LLM_NAME", "microsoft/Phi-4-mini-instruct")
VLLM_URL = os.getenv("vLLM_URL", "http://vllm:8000/v1/chat/completions")
VLLM_API_KEY = os.getenv("vLLM_API_KEY", "EMPTY")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 20))
FINAL_K = int(os.getenv("FINAL_K", 5))
STATIC_PROMPT = os.getenv("STATIC_PROMPT", "You are an expert on Dark Souls and Bloodborne.\n Answer ONLY using the provided context.\n\n Rules:\n - If the answer is not in the context, say: 'I don't know based on the provided context.'\n - Do not invent information.\n - Prefer specific details over vague summaries.\n - If multiple pieces of context conflict, mention that.\n")

logger = logging.getLogger(__name__)
embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(CROSS_ENCODER_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def rerank(query, chunks):
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:FINAL_K]]

async def retrieve_context(query: str) -> str:
    logger.info(f"Retrieving context for query: {query}")

    query_vector = (await asyncio.to_thread(embedder.encode, query)).tolist()

    hits = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=RETRIEVAL_K
    )

    chunks = [
        hit.payload["chunk_text"]
        for hit in hits.points
        if hit.payload
        and "chunk_text" in hit.payload
        and len(hit.payload["chunk_text"].strip()) > 50
    ]
    
    if not chunks:
        logger.warning(f"No relevant context found for query: {query}")
        return ""

    chunks = await asyncio.to_thread(rerank, query, chunks)
    chunks = chunks[:FINAL_K]

    logger.info(f"Retrieved {len(chunks)} context chunks for query: {query}")
    return "\n\n---\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    )

async def generate_streaming_response(request: ChatRequest):
    messages = request.messages
    user_query = messages[-1].content if messages else ""

    context = await retrieve_context(user_query)

    rag_messages = [
        {
            "role": "system",
            "content": STATIC_PROMPT
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"
        }
    ]

    new_request = {
        **request.model_dump(),
        "model": MODEL_NAME,
        "messages": rag_messages,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            VLLM_URL,
            headers={
                "Authorization": f"Bearer {VLLM_API_KEY}",
                "Accept": "text/event-stream"
            },
            json=new_request
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n"

async def generate_response(request: ChatRequest) -> Dict[str, Any]:
    user_query = request.messages[-1].content

    context = await retrieve_context(user_query)

    messages = [
        {
            "role": "system",
            "content": STATIC_PROMPT
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"
        }
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            VLLM_URL,
            headers={
                "Authorization": f"Bearer {VLLM_API_KEY}"
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        result = response.json()

    return result