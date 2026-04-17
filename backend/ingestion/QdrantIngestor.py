import os
import logging
from typing import List, Dict, Any
import hashlib

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams


logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks

class QdrantIngestor:
    def __init__(
        self,
        collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "soulsborne-collection"),
        model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model=None,
        host: str = os.getenv("QDRANT_HOST", "qdrant"),
        port: int = int(os.getenv("QDRANT_PORT", 6333)),
    ):
        # Embedding model
        self.model = model or SentenceTransformer(model_name)

        vector_size: int = self.model.get_sentence_embedding_dimension()

        # Qdrant client (Docker default)
        self.client = QdrantClient(host=host, port=port)

        self.collection_name = collection_name

        # Create collection if it doesn't exist
        self._ensure_collection(vector_size) 

    def _ensure_collection(self, vector_size: int):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def ingest_page(self, page: Dict[str, Any]):
        """
        page = {
            url,
            title,
            text,
            metadata
        }
        """

        logger.info(f"Chunking page: {page['url']}")

        chunks = chunk_text(page["text"])
        if not chunks:
            return 0
        
        logger.info(f"{len(chunks)} chunks created for page: {page['url']}")


        logger.info(f"Encoding embeddings for page: {page['url']}")
        vectors = self.model.encode(
            chunks, 
            show_progress_bar=False, 
            convert_to_numpy=True
        ).tolist()

        points = []

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = hashlib.md5(f"{page['url']}:{i}:{chunk}".encode()).hexdigest()

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "url": page["url"],
                        "title": page.get("title", ""),
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "metadata": {
                            **page.get("metadata", {}),
                            "chunk_length": len(chunk),
                            "source": "fextralife",
                        },
                    },
                )
            )

        BATCH_SIZE = 64

        logger.info(f"Upserting {len(chunks)} points for page: {page['url']}")

        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i+BATCH_SIZE],
                wait=True
            )

        logger.info(f"Completed ingestion for page: {page['url']}")

        return len(points)