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
from typing import Dict, List, Optional
from urllib.parse import urlparse, urljoin
from pydantic import HttpUrl # Add HttpUrl import
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

# Import configuration and schemas
from ..core.config import AppSettings
from ..core.schemas import ExtractionResult # Import from schemas

# Import custom exceptions
from ..core.exceptions import ScrapingError, ConfigurationError

# Import local modules for specialized scraping
from .scraping_utils import pdf, wikipedia

logger = logging.getLogger(__name__)

# Removed load_dotenv() call

# --- Removed local ExtractionResult dataclass --- #

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
                    download_pdfs = self.settings.scraper_download_pdfs
                    pdf_save_dir = self.settings.scraper_pdf_save_dir
                    content = await pdf.handle_pdf_url(url_str, download_pdfs, pdf_save_dir, max_pdf_size_bytes)
                    extraction_source = "pdf"
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
                # --- Add detailed logging for the passed crawler instance ---
                crawler_id = id(crawler) if crawler else None
                logger.debug(f"scrape() called for {url_str}. Passed crawler object ID: {crawler_id}. Is crawler None? {crawler is None}")
                # -------------------------------------------------------------
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

                    # --- Add logging just before the 'if crawler:' check ---
                    logger.debug(f"Checking 'if crawler:' condition for {url_str}. Passed crawler object ID: {crawler_id}. Is crawler None? {crawler is None}")
                    # -----------------------------------------------------
                    if crawler:
                        # Use the provided, already initialized crawler instance
                        logger.debug(f"Using provided crawler (ID: {crawler_id}) for {url_str}.")
                        await run_crawl_logic(crawler)
                    else:
                        # Create and manage a temporary crawler instance for this single scrape
                        logger.debug(f"Creating temporary AsyncWebCrawler for single URL: {url_str}.") # This line should ideally not be hit when called from scrape_many
                        async with AsyncWebCrawler(config=self.browser_config) as temp_crawler:
                            await run_crawl_logic(temp_crawler)

                    # --- Process results (using crawl4ai_result) ---
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
                            content = await pdf.handle_local_pdf_file(downloaded_pdf_path, max_size_bytes=max_pdf_size_bytes)
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
                        # Handle cases where crawl4ai_result is None or lacks content/markdown
                        if crawl4ai_result is None:
                             logger.warning(f"Crawl4AI processing failed or was skipped for {url_str}.")
                             extraction_source = "crawl4ai_failed_or_skipped"
                        else:
                             logger.warning(f"Crawl4AI processing for {url_str} yielded no usable content.")
                             extraction_source = "crawl4ai_empty_result"
                        content = None

                except Exception as e:
                    # Catch errors specifically within the general crawling dispatch logic itself (before arun)
                    logger.error(f"Unexpected error during general crawling setup for {url_str}: {e}", exc_info=True)
                    raise ScrapingError(f"General web crawling setup failed for {url_str}: {e}") from e

            # --- Final Result Construction --- #
            if content and content.strip():
                logger.info(f"Scraping successful for {url_str} using strategy: {extraction_source} (Content length: {len(content)})")
                return ExtractionResult(name=extraction_source, link=url_str, content=content)
            else:
                # Refined logging for extraction failure
                error_message = f"No content could be extracted by final strategy: {extraction_source}."
                if extraction_source == "pdf_download_processed_empty":
                    error_message += " PDF was downloaded but processing yielded no content."
                elif extraction_source == "crawl4ai_failed_or_skipped":
                     error_message += " Crawl4AI execution failed or was skipped due to earlier errors."
                elif extraction_source == "crawl4ai_empty_result":
                     error_message += " Crawl4AI executed but returned no usable content or markdown."


                logger.warning(f"Extraction failed for URL {url_str}: {error_message}")
                return ExtractionResult(name=extraction_source, link=url_str, content=None, error=f"No content extracted ({extraction_source})")

        except (ValueError, ScrapingError, ConfigurationError) as e:
             logger.error(f"Scraping failed for {url_str}: {type(e).__name__}: {e}", exc_info=False)
             # Return an ExtractionResult with error status instead of raising here
             # This allows scrape_many to collect individual errors
             return ExtractionResult(name="scraping_error", link=url_str, content=None, error=f"{type(e).__name__}: {e}")
        except Exception as e:
            logger.critical(f"Critical unexpected error during scrape() for {url_str}: {e}", exc_info=True)
            # Return an ExtractionResult with error status
            return ExtractionResult(name="unexpected_critical_error", link=url_str, content=None, error=f"Unexpected critical error: {e}")

    # --- Refactored scrape_many ---
    async def scrape_many(
        self,
        urls: List[str],
        sequential: bool = False # Default back to False, allowing concurrency
    ) -> Dict[str, ExtractionResult]:
        if not urls:
            return {}

        results: Dict[str, ExtractionResult] = {}
        total_urls = len(urls)

        # Create a single crawler instance to be reused
        async with AsyncWebCrawler(config=self.browser_config) as shared_crawler:
            if sequential:
                logger.info(f"Processing {total_urls} URLs sequentially using shared crawler...")
                for i, url in enumerate(urls):
                    logger.debug(f"Scraping URL {i+1}/{total_urls} (sequential): {url}")
                    # Call scrape, passing the shared crawler instance
                    results[url] = await self.scrape(url, crawler=shared_crawler)
            else:
                logger.info(f"Processing {total_urls} URLs concurrently using shared crawler...")
                # Create tasks, passing the shared crawler instance to each scrape call
                tasks = {url: asyncio.create_task(self.scrape(url, crawler=shared_crawler)) for url in urls}

                task_results = await asyncio.gather(*tasks.values()) # No return_exceptions needed if scrape handles them

                url_list = list(tasks.keys())
                for i, result_or_exc in enumerate(task_results):
                    url = url_list[i]
                    # Since scrape now returns ExtractionResult even on error, just assign it
                    if isinstance(result_or_exc, ExtractionResult):
                         results[url] = result_or_exc
                    else: # Should not happen if scrape is correctly implemented
                         logger.error(f"Unexpected return type from scrape task for URL '{url}': {type(result_or_exc)}")
                         results[url] = ExtractionResult(name="internal_error", link=url, content=None, error=f"Unexpected return type: {type(result_or_exc)}")


        # Post-processing: Log summary of errors encountered
        exceptions_count = sum(1 for res in results.values() if res.error)
        if exceptions_count > 0:
            logger.warning(f"Completed scraping batch for {total_urls} URLs with {exceptions_count} failures.")
            # Optionally log specific failed URLs
            for url, res in results.items():
                 if res.error:
                      logger.debug(f"  - Failed URL: {url} (Error: {res.error})")
        else:
             logger.info(f"Successfully completed scraping batch for {total_urls} URLs.")

        return results

# Removed old helper methods (_fetch_raw_html, extract)
# Removed old classes/functions (StrategyFactory, ExtractionConfig, HTML cleaners, is_academic_url)

# Example usage can be added here if needed
# if __name__ == "__main__":
#     async def main():
#         # ... example setup and calls ...
#         pass
#     asyncio.run(main()) 