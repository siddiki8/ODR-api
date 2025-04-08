"""
Service for scraping web content using Crawl4AI and specialized handlers.

Provides a unified interface (`WebScraper`) to extract content from various URLs,
including general web pages, Wikipedia pages, and PDFs (direct links or downloaded).
"""

import asyncio
import logging
import os # Needed for path operations with downloaded files
# Removed unused: import re
# Removed unused dataclass: from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from urllib.parse import urlparse, urljoin
from pydantic import BaseModel, Field, ConfigDict, HttpUrl # Add necessary Pydantic imports
import httpx # <-- Import httpx for exception handling
import fitz # PyMuPDF for PDF handling

# External Dependencies
# Removed unused: import httpx
# Removed unused: from dotenv import load_dotenv

# Crawl4AI Imports
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MarkdownGenerationResult,
    CrawlResult # Import CrawlResult for type hinting
)

# Import configuration and schemas using absolute paths
from app.core.config import AppSettings
# --- REMOVED core schema import --- 
# from app.core.schemas import ExtractionResult 

# Import custom exceptions using absolute paths
from app.core.exceptions import ScrapingError, ConfigurationError

# Import local modules for specialized scraping (update relative path)
# Assumes scraper_utils is moved alongside this file
from .scraper_utils import wikipedia 
# Removed reference to non-existent pdf module

logger = logging.getLogger(__name__)

# Removed load_dotenv() call

# --- ExtractionResult Schema Definition --- #
class ExtractionResult(BaseModel):
    """Represents the outcome of a content extraction attempt for a single URL."""
    model_config = ConfigDict(extra='ignore') # Ignore extra fields if any during validation

    source_url: HttpUrl = Field(..., description="The URL that was processed.")
    content: Optional[str] = Field(None, description="The extracted text content. None if extraction failed or yielded no content.")
    extraction_source: str = Field("unknown", description="Identifier for the source type or extraction method (e.g., 'crawl4ai', 'wikipedia', 'pdf_direct').")
    error_message: Optional[str] = Field(None, description="Error message if extraction failed.")
    status: Literal['success', 'empty', 'error'] = Field(..., description="Indicates the outcome: 'success' (content found), 'empty' (no content found/extracted), 'error' (an exception occurred).")

# --- Helper Function for Local PDF Handling (Moved from non-existent pdf.py) --- #
async def handle_local_pdf_file(file_path: str, max_size_bytes: Optional[int] = None) -> str:
    """Extracts text content from a local PDF file.

    Args:
        file_path: Path to the local PDF file.
        max_size_bytes: Optional maximum file size in bytes.

    Returns:
        Extracted text content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ScrapingError: If the file is too large or cannot be processed by PyMuPDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local PDF file not found: {file_path}")

    # Check size limit
    if max_size_bytes is not None:
        file_size = os.path.getsize(file_path)
        if file_size > max_size_bytes:
            logger.warning(f"Local PDF {file_path} ({file_size} bytes) exceeds max size ({max_size_bytes} bytes). Skipping.")
            raise ScrapingError(f"Local PDF exceeds max size limit ({max_size_bytes / (1024*1024):.1f} MB)")

    try:
        logger.debug(f"Opening local PDF: {file_path}")
        doc = fitz.open(file_path)
        text_content = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text("text") + "\n\n" # Add space between pages
        doc.close()
        logger.debug(f"Successfully extracted text from local PDF: {file_path}")
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error processing local PDF file {file_path} with PyMuPDF: {e}", exc_info=True)
        raise ScrapingError(f"Failed to process local PDF file {file_path}: {e}") from e

# --- WebScraper Class --- #
class WebScraper:
    """
    Coordinates web scraping using specialized handlers (Wikipedia, PDF)
    or the general-purpose Crawl4AI library.

    Uses AppSettings for configuration (PDF saving, size limits, etc.).
    Can handle directly linked PDFs and PDFs downloaded via Crawl4AI.
    """

    def __init__(
        self,
        settings: AppSettings,
        debug: bool = False,
        browser_config: Optional[BrowserConfig] = None
    ):
        """Initializes the WebScraper.

        Args:
            settings: The application settings instance.
            debug: If True, enables verbose logging for Crawl4AI browser operations.
            browser_config: Optional custom BrowserConfig for Crawl4AI. If provided,
                            ensure it's configured for downloads if needed.
        """
        self.settings = settings
        self.debug = debug

        # Configure browser settings for Crawl4AI
        default_download_path = os.path.join(settings.scraper_pdf_save_dir, ".crawl4ai_dl")
        os.makedirs(default_download_path, exist_ok=True) # Ensure dir exists

        if browser_config:
            self.browser_config = browser_config
            if not getattr(self.browser_config, 'accept_downloads', False):
                 logger.warning("Custom BrowserConfig provided but accept_downloads is not True. Crawl4AI may not download linked PDFs.")
        else:
            self.browser_config = BrowserConfig(
                browser_type='chromium',
                headless=True,
                verbose=self.debug,
                accept_downloads=True, # Enable downloads
                downloads_path=default_download_path # Specify download location
            )

        logger.info(
            f"WebScraper initialized. Debug: {self.debug}, "
            f"Browser Type: {self.browser_config.browser_type}, "
            f"Accept Downloads: {getattr(self.browser_config, 'accept_downloads', False)}, "
            f"Downloads Path: {getattr(self.browser_config, 'downloads_path', 'Default')}"
        )

    async def _handle_direct_pdf_url(self, url: str, max_size_bytes: Optional[int]) -> str:
        """Handles scraping directly linked PDF URLs."""
        logger.debug(f"Handling direct PDF URL: {url}")
        temp_pdf_path = None
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                async with client.stream('GET', url) as response:
                    response.raise_for_status() # Check for initial errors

                    # Check content type if available
                    content_type = response.headers.get('content-type', '').lower()
                    if 'application/pdf' not in content_type:
                        logger.warning(f"URL {url} Content-Type is not PDF ('{content_type}'). Skipping direct PDF handling.")
                        raise ScrapingError(f"URL Content-Type is not application/pdf")

                    # Check size limit before downloading fully
                    content_length = response.headers.get('content-length')
                    if content_length and max_size_bytes is not None:
                        if int(content_length) > max_size_bytes:
                            raise ScrapingError(f"Direct PDF URL content-length ({content_length}) exceeds max size ({max_size_bytes})")

                    # Download to temp file (or memory)
                    # Using a temporary file is safer for large files
                    # TODO: Consider using tempfile module for better management
                    temp_pdf_path = os.path.join(self.settings.scraper_pdf_save_dir, f"temp_{os.path.basename(urlparse(url).path)}")
                    os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)

                    downloaded_size = 0
                    with open(temp_pdf_path, 'wb') as f:
                        async for chunk in response.aiter_bytes():
                            downloaded_size += len(chunk)
                            if max_size_bytes is not None and downloaded_size > max_size_bytes:
                                raise ScrapingError(f"Direct PDF URL download exceeded max size ({max_size_bytes}) during streaming")
                            f.write(chunk)

            logger.info(f"Downloaded direct PDF from {url} to {temp_pdf_path}")
            # Process the downloaded temp file
            content = await handle_local_pdf_file(temp_pdf_path, max_size_bytes=None) # Already checked size
            return content
        except httpx.HTTPStatusError as e:
            # Log 403 Forbidden specifically as a warning, other HTTP errors as errors
            if e.response.status_code == 403:
                logger.warning(f"Access Forbidden (403) fetching direct PDF URL {url}. Might require authentication or have access restrictions.")
            else:
                logger.error(f"HTTP Error {e.response.status_code} fetching direct PDF URL {url}")
            # Still raise ScrapingError as the fetch failed
            raise ScrapingError(f"HTTP Error {e.response.status_code} fetching direct PDF URL {url}") from e
        except httpx.RequestError as e:
            raise ScrapingError(f"Network Error fetching direct PDF URL {url}: {e}") from e
        finally:
            # Clean up temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    logger.debug(f"Removed temporary PDF file: {temp_pdf_path}")
                except OSError as e:
                    logger.error(f"Error removing temporary PDF file {temp_pdf_path}: {e}")


    async def scrape(self, url: str, crawler: Optional[AsyncWebCrawler] = None) -> ExtractionResult:
        """
        Scrapes content from a single URL using specialized handlers or Crawl4AI.

        Args:
            url: The URL to scrape.
            crawler: An optional existing AsyncWebCrawler instance to reuse for general crawling.
                     If None, a temporary one will be created if needed.

        Returns:
            An ExtractionResult Pydantic model.
        """
        url_str = str(url) if isinstance(url, HttpUrl) else url

        if not url_str or not urlparse(url_str).scheme in ['http', 'https']:
            raise ValueError(f"Invalid URL provided for scraping: {url_str}")

        parsed_url_obj = urlparse(url_str)
        path_lower = parsed_url_obj.path.lower()
        is_pdf_link = path_lower.endswith('.pdf')

        content: Optional[str] = None
        extraction_source = "unknown"
        max_pdf_size_bytes = self.settings.scraper_max_pdf_size_mb * 1024 * 1024

        try:
            # --- URL Dispatching --- #

            # 1. Wikipedia Handler
            if "wikipedia.org" in parsed_url_obj.netloc:
                try:
                    logger.info(f"Dispatching to Wikipedia handler for: {url_str}")
                    content = await wikipedia.get_wikipedia_content(url_str)
                    extraction_source = "wikipedia"
                    logger.info(f"Wikipedia handler successful for: {url_str}")
                except ScrapingError as e:
                    logger.error(f"Wikipedia extraction failed for {url_str}: {e}", exc_info=False)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in Wikipedia handler for {url_str}: {e}", exc_info=True)
                    raise ScrapingError(f"Wikipedia extraction failed unexpectedly for {url_str}: {e}") from e

            # 2. Direct PDF Handler
            elif is_pdf_link:
                try:
                    logger.info(f"Dispatching to PDF handler for direct URL: {url_str}")
                    content = await self._handle_direct_pdf_url(url_str, max_pdf_size_bytes)
                    extraction_source = "pdf_direct"
                    logger.info(f"PDF handler successful for direct URL: {url_str}")
                except ScrapingError as e:
                    logger.error(f"PDF handling failed for direct URL {url_str}: {e}", exc_info=False)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in PDF handler for direct URL {url_str}: {e}", exc_info=True)
                    raise ScrapingError(f"PDF handling failed unexpectedly for direct URL {url_str}: {e}") from e

            # 3. General Web Crawling with Crawl4AI
            else:
                logger.info(f"Dispatching to general web crawler (Crawl4AI): {url_str}")
                crawler_id = id(crawler) if crawler else None
                logger.debug(f"scrape() called for {url_str}. Passed crawler object ID: {crawler_id}. Is crawler None? {crawler is None}")
                try:
                    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                    crawl4ai_result: Optional[CrawlResult] = None

                    async def run_crawl_logic(crawler_instance: AsyncWebCrawler):
                        nonlocal crawl4ai_result
                        instance_id = id(crawler_instance)
                        logger.debug(f"run_crawl_logic executing for {url_str} with crawler instance ID: {instance_id}")
                        try:
                            crawl4ai_result = await crawler_instance.arun(url=url_str, config=run_config)
                            logger.debug(f"run_crawl_logic completed arun for {url_str} with crawler instance ID: {instance_id}")
                        except (httpx.HTTPStatusError, httpx.RequestError) as e:
                            status_code = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else "N/A"
                            logger.warning(f"Crawl4AI HTTP/Network Error for {url_str} (Status: {status_code}, Crawler ID: {instance_id}): {e}")
                        except Exception as e:
                            logger.error(f"Crawl4AI encountered an unexpected error during arun for {url_str} (Crawler ID: {instance_id}): {e}", exc_info=True)

                    logger.debug(f"Checking 'if crawler:' condition for {url_str}. Passed crawler object ID: {crawler_id}. Is crawler None? {crawler is None}")
                    if crawler:
                        logger.debug(f"Using provided crawler (ID: {crawler_id}) for {url_str}.")
                        await run_crawl_logic(crawler)
                    else:
                        logger.debug(f"Creating temporary AsyncWebCrawler for single URL: {url_str}.")
                        async with AsyncWebCrawler(config=self.browser_config) as temp_crawler:
                            await run_crawl_logic(temp_crawler)

                    # Process results
                    downloaded_pdf_path: Optional[str] = None
                    if crawl4ai_result and hasattr(crawl4ai_result, 'downloaded_files') and crawl4ai_result.downloaded_files:
                        downloaded_files_list = crawl4ai_result.downloaded_files if isinstance(crawl4ai_result.downloaded_files, list) else []
                        for file_path in downloaded_files_list:
                            if isinstance(file_path, str) and file_path.lower().endswith('.pdf'):
                                if os.path.exists(file_path):
                                    downloaded_pdf_path = file_path
                                    logger.info(f"Found downloaded PDF via Crawl4AI: {file_path}")
                                    break

                    if downloaded_pdf_path:
                        logger.info(f"Processing downloaded PDF from Crawl4AI using handle_local_pdf_file: {downloaded_pdf_path}")
                        try:
                            content = await handle_local_pdf_file(downloaded_pdf_path, max_size_bytes=max_pdf_size_bytes)
                            extraction_source = "pdf_downloaded_by_crawl4ai"
                            if content:
                                logger.info(f"Successfully extracted text from downloaded PDF: {downloaded_pdf_path}")
                            else:
                                logger.warning(f"handle_local_pdf_file returned no content for downloaded PDF: {downloaded_pdf_path}")
                                extraction_source = "pdf_download_processed_empty"
                        except FileNotFoundError:
                             logger.error(f"Downloaded PDF not found at expected path: {downloaded_pdf_path}")
                             content = None
                             extraction_source = "pdf_download_not_found"
                        except ScrapingError as e:
                             logger.error(f"Failed to process downloaded PDF {downloaded_pdf_path} with handle_local_pdf_file: {e}", exc_info=False)
                             content = None
                             extraction_source = "pdf_download_error"
                        except Exception as e:
                             logger.error(f"Unexpected error processing downloaded PDF {downloaded_pdf_path} with handle_local_pdf_file: {e}", exc_info=True)
                             content = None
                             extraction_source = "pdf_download_unexpected_error"

                    elif crawl4ai_result and hasattr(crawl4ai_result, 'content') and crawl4ai_result.content:
                        content = crawl4ai_result.content
                        extraction_source = crawl4ai_result.name if hasattr(crawl4ai_result, 'name') else "crawl4ai_content"
                    elif crawl4ai_result and hasattr(crawl4ai_result, 'markdown') and crawl4ai_result.markdown:
                        content = crawl4ai_result.markdown
                        extraction_source = crawl4ai_result.name if hasattr(crawl4ai_result, 'name') else "crawl4ai_markdown"
                    else:
                        if crawl4ai_result is None:
                             logger.warning(f"Crawl4AI processing failed or was skipped for {url_str}.")
                             extraction_source = "crawl4ai_failed_or_skipped"
                        else:
                             logger.warning(f"Crawl4AI processing for {url_str} yielded no usable content.")
                             extraction_source = "crawl4ai_empty_result"
                        content = None

                except Exception as e:
                    logger.error(f"Unexpected error during general crawling setup for {url_str}: {e}", exc_info=True)
                    raise ScrapingError(f"General web crawling setup failed for {url_str}: {e}") from e

            # Final Result Construction
            if content is None:
                logger.warning(f"Scraping {url_str} resulted in no content (Source: {extraction_source}).")
            else:
                logger.info(f"Successfully scraped content from {url_str} (Source: {extraction_source}, Length: {len(content)})")

            return ExtractionResult(
                content=content,
                source_url=url_str,
                status="success" if content is not None else "empty",
                extraction_source=extraction_source
            )

        except Exception as e:
            logger.error(f"Scraping failed for URL: {url_str}. Error: {e}", exc_info=True)
            return ExtractionResult(
                content=None,
                source_url=url_str,
                status="error",
                error_message=str(e),
                extraction_source=extraction_source # Record where it failed
            )

    async def scrape_many(
        self,
        urls: List[str],
        sequential: bool = False # Default back to False, allowing concurrency
    ) -> Dict[str, ExtractionResult]:
        """
        Scrapes multiple URLs, potentially concurrently using a shared Crawl4AI instance.

        Args:
            urls: A list of URLs to scrape.
            sequential: If True, scrape URLs one by one. If False (default), use concurrency.

        Returns:
            A dictionary mapping each URL to its ExtractionResult.
        """
        results: Dict[str, ExtractionResult] = {}
        if not urls:
            return results

        if sequential:
            logger.info(f"Starting sequential scraping for {len(urls)} URLs.")
            for url in urls:
                # Create a temporary crawler for each URL in sequential mode
                results[url] = await self.scrape(url, crawler=None)
        else:
            logger.info(f"Starting concurrent scraping for {len(urls)} URLs.")
            # Use a shared crawler instance for concurrent runs
            async with AsyncWebCrawler(config=self.browser_config) as shared_crawler:
                tasks = [self.scrape(url, crawler=shared_crawler) for url in urls]
                # This gathers results OR exceptions
                scrape_results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

            # Process the results from asyncio.gather
            for i, result_or_error in enumerate(scrape_results_or_errors):
                original_url = urls[i] # Get the URL corresponding to this result
                if isinstance(result_or_error, ExtractionResult):
                    results[original_url] = result_or_error
                    # Log success/empty status from the result itself
                    if result_or_error.status == 'success':
                        logger.debug(f"Concurrent scrape successful for {original_url}")
                    elif result_or_error.status == 'empty':
                        logger.warning(f"Concurrent scrape yielded empty content for {original_url}")
                elif isinstance(result_or_error, Exception):
                    logger.error(f"Concurrent scrape failed for {original_url}: {result_or_error}", exc_info=False) # exc_info=False as gather already captured it
                    # Create an error ExtractionResult
                    results[original_url] = ExtractionResult(
                        source_url=original_url,
                        status="error",
                        error_message=f"Scraping failed: {str(result_or_error)}",
                        extraction_source="concurrent_gather_exception"
                    )
                else:
                    # Should not happen with return_exceptions=True, but handle defensively
                    logger.error(f"Unexpected item type returned from asyncio.gather for {original_url}: {type(result_or_error)}")
                    results[original_url] = ExtractionResult(
                        source_url=original_url,
                        status="error",
                        error_message=f"Unexpected result type: {type(result_or_error)}",
                        extraction_source="concurrent_gather_unexpected_type"
                    )

        logger.info(f"Finished scraping {len(urls)} URLs.")
        return results 