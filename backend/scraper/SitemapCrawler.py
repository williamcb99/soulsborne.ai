import asyncio
import importlib
import json
import logging
from typing import Any, Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_SITEMAP_URL = "https://bloodborne.wiki.fextralife.com/sitemap.xml"
DEFAULT_DROP_DIV_CLASSES = ["discussion-wrapper", "an-rail", "an-injected", "embed-responsive"]
DEFAULT_DROP_DIV_ID = ["featured-wikis-container"]


_TRAFILATURA: Optional[Any]
try:
    _TRAFILATURA = importlib.import_module("trafilatura")
except Exception:
    _TRAFILATURA = None


class SitemapCrawler:
    def __init__(
        self,
        sitemap_url: str = DEFAULT_SITEMAP_URL,
        *,
        user_agent: str = "SoulsborneBot/1.0 (+https://example.local; educational scraper)",
        timeout_seconds: float = 20.0,
        retries: int = 3,
        delay_seconds: float = 0.5,
        concurrency: int = 5,
        max_sitemaps: int = 500,
        default_max_pages: int = 100,
        drop_div_classes: Optional[List[str]] = None,
        drop_div_ids: Optional[List[str]] = None,
        use_trafilatura: bool = True,
    ) -> None:
        self.sitemap_url = sitemap_url
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.delay_seconds = delay_seconds
        self.concurrency = concurrency
        self.max_sitemaps = max_sitemaps
        self.default_max_pages = default_max_pages
        self.drop_div_classes = drop_div_classes or list(DEFAULT_DROP_DIV_CLASSES)
        self.drop_div_ids = drop_div_ids or list(DEFAULT_DROP_DIV_ID)
        self.use_trafilatura = use_trafilatura

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    async def _fetch_text_with_retries(self, client: httpx.AsyncClient, url: str) -> str:
        """Fetch text content with retry/backoff for transient failures."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                response = await client.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.text
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")

    @staticmethod
    def parse_sitemap_xml(xml_text: str) -> Dict[str, List[str]]:
        """
        Return sitemap links and page links from a sitemap XML document.

        Supports both:
        - sitemapindex: <sitemap><loc>...</loc></sitemap>
        - urlset: <url><loc>...</loc></url>
        """
        root = ET.fromstring(xml_text)

        sitemap_links: List[str] = []
        page_links: List[str] = []

        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0].strip("{")
            ns = {"sm": namespace}
            sitemap_links = [
                loc.text.strip()
                for loc in root.findall(".//sm:sitemap/sm:loc", ns)
                if loc.text
            ]
            page_links = [
                loc.text.strip() for loc in root.findall(".//sm:url/sm:loc", ns) if loc.text
            ]
        else:
            sitemap_links = [
                loc.text.strip() for loc in root.findall(".//sitemap/loc") if loc.text
            ]
            page_links = [loc.text.strip() for loc in root.findall(".//url/loc") if loc.text]

        return {"sitemaps": sitemap_links, "pages": page_links}

    async def gather_urls_from_sitemap(
        self,
        client: httpx.AsyncClient,
        sitemap_url: Optional[str] = None,
    ) -> List[str]:
        """Recursively collect all page URLs referenced by a sitemap or sitemap index."""
        start_sitemap = sitemap_url or self.sitemap_url
        visited_sitemaps: Set[str] = set()
        queued_sitemaps: List[str] = [start_sitemap]
        pages: Set[str] = set()

        while queued_sitemaps and len(visited_sitemaps) < self.max_sitemaps:
            current = queued_sitemaps.pop(0)
            if current in visited_sitemaps:
                continue

            visited_sitemaps.add(current)
            xml_text = await self._fetch_text_with_retries(client, current)
            parsed = self.parse_sitemap_xml(xml_text)

            for nested in parsed["sitemaps"]:
                if nested not in visited_sitemaps:
                    queued_sitemaps.append(nested)

            for page in parsed["pages"]:
                pages.add(page)

        return sorted(pages)

    def _extract_with_trafilatura(self, html: str, url: str) -> Optional[Dict]:
        if not self.use_trafilatura or _TRAFILATURA is None:
            return None

        result = _TRAFILATURA.extract(
            html,
            url=url,
            output_format="json",
            include_links=False,
            include_tables=False,
            favor_precision=True,
        )

        if not result:
            logger.warning(f"trafilatura failed for {url}")
            return None
        try:
            return json.loads(result)
        except Exception:
            logger.warning(f"failed to parse trafilatura result for {url}")
            return None

    def _is_valid_text(self, text: str) -> bool:
        words = text.split()
        return (
            len(words) > 40
            and len(set(words)) / max(len(words), 1) > 0.3
        )

    def extract_page_data(self, html: str, url: str) -> Dict[str, str]:
        """Extract structured fields from an HTML page."""
        data = self._extract_with_trafilatura(html, url)
        
        if data and self._is_valid_text(data.get("text", "")):
            return {
                "url": url,
                "title": data.get("title", ""),
                "text": data.get("text", ""),
                "metadata": {
                    "author": data.get("author"),
                    "date": data.get("date"),
                    "description": data.get("description"),
                    "source": "fextralife"
                }
            }
        
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        meta_description = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_description = str(meta_tag["content"]).strip()
        for tag in soup(["script", "style", "noscript", "nav", "form", "footer", "img", "iframe"]):
            tag.decompose()

        for class_name in self.drop_div_classes:
            for tag in soup.select(f"div.{class_name}"):
                tag.decompose()

        for div_id in self.drop_div_ids:
            for tag in soup.select(f"div#{div_id}"):
                tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        return {
                "url": url,
                "title": title,
                "text": text,
                "metadata": {
                    "author": "",
                    "date": "",
                    "description": meta_description,
                    "source": "fextralife"
                }
            }

    async def _crawl_one(
        self,
        client: httpx.AsyncClient,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Dict[str, str]]:
        async with semaphore:
            try:
                html = await self._fetch_text_with_retries(client, url)
                return self.extract_page_data(html, url)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error occurred while crawling {url}: {exc}")
                return None
            finally:
                if self.delay_seconds > 0:
                    await asyncio.sleep(self.delay_seconds)

    async def crawl(
        self,
        *,
        sitemap_url: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Discover URLs from sitemap and asynchronously crawl page content."""
        page_limit = max_pages if max_pages is not None else self.default_max_pages

        async with httpx.AsyncClient(follow_redirects=True, headers=self.headers) as client:
            urls = await self.gather_urls_from_sitemap(client, sitemap_url=sitemap_url)
            target_urls = urls[:page_limit]

            semaphore = asyncio.Semaphore(self.concurrency)
            tasks = [
                self._crawl_one(client, url, semaphore)
                for url in target_urls
            ]

            results: List[Dict[str, str]] = []
            for item in await asyncio.gather(*tasks):
                if item is not None:
                    results.append(item)
            return results
