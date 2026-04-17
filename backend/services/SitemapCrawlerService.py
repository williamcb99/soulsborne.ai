from scraper.SitemapCrawler import SitemapCrawler
from ingestion.QdrantIngestor import QdrantIngestor
from typing import Optional
from classes.IngestRequest import IngestRequest
import time
import logging


logger = logging.getLogger(__name__)

async def run_ingestion(
        request: Optional[IngestRequest] = None
        ):
    request = request or IngestRequest()
    scraper = SitemapCrawler()
    ingestor = QdrantIngestor()

    logger.info(f"Starting crawl at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    pages = await scraper.crawl(max_pages=request.max_pages, sitemap_url=request.sitemap_url)

    logger.info(f"Crawl completed with {len(pages)} pages at {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    for i, page in enumerate(pages):
        logger.info(f"Ingesting page {i+1}/{len(pages)}: {page['url']}")
        ingestor.ingest_page(page)