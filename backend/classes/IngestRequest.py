from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    sitemap_url: Optional[str] = None
    max_pages: Optional[int] = None