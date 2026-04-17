from typing import Any, Dict, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import uuid
from classes.IngestRequest import IngestRequest
from classes.ChatRequest import ChatRequest
from services.SitemapCrawlerService import run_ingestion
from services.QueryService import generate_response, generate_streaming_response
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks, 
    request: Optional[IngestRequest] = None
    ):
    """
    Ingest pages from a sitemap URL into Qdrant. Defaults to Bloodborne wiki if no URL provided and limits to 100 pages.
    """
    if request and request.max_pages <= 0:
        return {"status": "error", "message": "max_pages must be a positive integer"}
    
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_ingestion, request)
    return {"status": "started", "job_id": job_id}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> Dict[str, Any]:
    if request.stream:
        return StreamingResponse(
            generate_streaming_response(request),
            media_type="text/event-stream"
        )
    
    response = await generate_response(request)
    
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "choices": [
             {
                "index": 0,
                "message": response["choices"][0]["message"],
                "finish_reason": "stop",
            }
        ],
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "soulsborne.ai",
                "object": "model",
                "owned_by": "local",
                "permission": [],
                "root": "soulsborne.ai",
                "parent": None
            }
        ]
    }