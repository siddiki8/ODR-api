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
        # Enable downloads by default using settings
        default_download_path = os.path.join(settings.scraper_pdf_save_dir, ".crawl4ai_dl")
        os.makedirs(default_download_path, exist_ok=True) # Ensure dir exists
        
        if browser_config:
            # If user provides a config, use it but warn if downloads are off
            self.browser_config = browser_config
            if not getattr(self.browser_config, 'accept_downloads', False):
                 logger.warning("Custom BrowserConfig provided but accept_downloads is not True. "
                                "Crawl4AI may not download linked PDFs.")
        else:
            # Default config: enable downloads, use path from settings
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

    async def scrape(self, url: str) -> ExtractionResult:
        """
        Scrapes content from a single URL.

        Handles direct PDF links, Wikipedia pages, and general web pages.
        For general pages, it uses Crawl4AI, checks if a PDF was downloaded,
        and processes either the downloaded PDF or the extracted Markdown.

        Args:
            url: The URL to scrape.

        Returns:
            An ExtractionResult Pydantic model containing the content and source name.

        Raises:
            ValueError: If the URL is invalid.
            ScrapingError: If extraction fails for any reason.
            ConfigurationError: If required configuration is missing.
        """
        # Ensure URL is a string *before* any parsing
        url_str = str(url) if isinstance(url, HttpUrl) else url

        if not url_str or not urlparse(url_str).scheme in ['http', 'https']:
            raise ValueError(f"Invalid URL provided for scraping: {url_str}")

        parsed_url_obj = urlparse(url_str)
        content: Optional[str] = None
        extraction_source = "unknown"
        max_pdf_size_bytes = self.settings.scraper_max_pdf_size_mb * 1024 * 1024

        try:
            # --- URL Dispatching --- #

            # 1. Wikipedia Handler
            if "wikipedia.org" in parsed_url_obj.netloc:
                try:
                    logger.info(f"Dispatching to Wikipedia handler for: {url}")
                    content = await wikipedia.get_wikipedia_content(url)
                    extraction_source = "wikipedia"
                    logger.info(f"Wikipedia handler successful for: {url}")
                except ScrapingError as e:
                    logger.error(f"Wikipedia extraction failed for {url}: {e}", exc_info=False)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in Wikipedia handler for {url}: {e}", exc_info=True)
                    raise ScrapingError(f"Wikipedia extraction failed unexpectedly for {url}: {e}") from e

            # 2. Direct PDF Handler
            elif url_str.lower().endswith('.pdf'):
                try:
                    logger.info(f"Dispatching to PDF handler for direct URL: {url_str}")
                    download_pdfs = self.settings.scraper_download_pdfs # Keep original file?
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
                try:
                    # Define crawl4ai configuration
                    # Use CacheMode.NORMAL during testing/dev, BYPASS for production maybe?
                    # Cache path could be configurable
                    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                    # Use browser config from __init__ (already configured for downloads)
                    async with AsyncWebCrawler(config=self.browser_config) as crawler:
                        # --- Wrap crawler.arun in try/except --- #
                        crawl4ai_result: Optional[ExtractionResult] = None # Default to None
                        try:
                            crawl4ai_result = await crawler.arun(url=url_str, config=run_config)
                        except (httpx.HTTPStatusError, httpx.RequestError) as e:
                             # Handle HTTP/Network errors from underlying httpx call within Crawl4AI
                             status_code = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else "N/A"
                             logger.warning(f"Crawl4AI failed for {url_str}: HTTP/Network Error (Status: {status_code}) - {e}")
                             # crawl4ai_result remains None
                        except Exception as e:
                             # Catch any other unexpected errors during crawler run
                             logger.error(f"Crawl4AI encountered an unexpected error for {url_str}: {e}", exc_info=True)
                             # crawl4ai_result remains None
                        # ----------------------------------------- #

                    # --- Check for Downloaded PDF --- 
                    downloaded_pdf_path: Optional[str] = None
                    if crawl4ai_result and hasattr(crawl4ai_result, 'downloaded_files') and crawl4ai_result.downloaded_files:
                        # Ensure downloaded_files is iterable and contains strings
                        downloaded_files_list = crawl4ai_result.downloaded_files if isinstance(crawl4ai_result.downloaded_files, list) else []
                        for file_path in downloaded_files_list:
                            # Check if a downloaded file looks like a PDF
                            if isinstance(file_path, str) and file_path.lower().endswith('.pdf'):
                                if os.path.exists(file_path):
                                     downloaded_pdf_path = file_path
                                     logger.info(f"Found downloaded PDF via Crawl4AI: {file_path}")
                                     break # Use the first found PDF

                    # --- Process results --- #
                    if downloaded_pdf_path:
                        # If Crawl4AI downloaded a PDF, process it directly
                        logger.info(f"Processing downloaded PDF from Crawl4AI: {downloaded_pdf_path}")
                        try:
                            # Re-use PDF handling logic, adjust as needed for local file path
                            # Assuming a function handle_local_pdf_file exists or adapt handle_pdf_url
                            # For now, let's assume PyMuPDF extraction similar to handle_pdf_url
                            # --- PyMuPDF Extraction for Local File --- #
                            pdf_document = fitz.open(downloaded_pdf_path)
                            all_text = []
                            for page_num in range(len(pdf_document)):
                                page = pdf_document.load_page(page_num)
                                all_text.append(page.get_text("text", sort=True))
                            content = "\n".join(all_text).strip()
                            content = content.replace("\t", " ")
                            content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
                            pdf_document.close()
                            extraction_source = "pdf_downloaded_by_crawl4ai"
                            logger.info(f"Successfully extracted text from downloaded PDF: {downloaded_pdf_path}")
                            # --- End PyMuPDF --- #
                        except Exception as e:
                            logger.error(f"Failed to process downloaded PDF {downloaded_pdf_path}: {e}", exc_info=True)
                            content = None # Failed processing means no content
                            extraction_source = "pdf_download_error"

                    elif crawl4ai_result and hasattr(crawl4ai_result, 'content') and crawl4ai_result.content:
                         content = crawl4ai_result.content # Use general content if available
                         extraction_source = crawl4ai_result.name if hasattr(crawl4ai_result, 'name') else "crawl4ai_content"
                    elif crawl4ai_result and hasattr(crawl4ai_result, 'markdown') and crawl4ai_result.markdown:
                        content = crawl4ai_result.markdown # Fallback to markdown
                        extraction_source = crawl4ai_result.name if hasattr(crawl4ai_result, 'name') else "crawl4ai_markdown"
                    else:
                        # This case is hit if crawl4ai_result is None (due to exceptions caught above)
                        # or if the result object exists but has no content/markdown.
                        logger.warning(f"Crawl4AI processing for {url_str} yielded no usable content or failed.")
                        content = None # Ensure content is None if no usable output
                        extraction_source = "crawl4ai_failed_or_empty"

                except Exception as e:
                    # Catch errors specifically within the general crawling block
                    logger.error(f"Unexpected error during general crawling dispatch for {url_str}: {e}", exc_info=True)
                    raise ScrapingError(f"General web crawling failed for {url_str}: {e}") from e

            # --- Final Result Construction --- #
            if content and content.strip():
                logger.info(f"Scraping successful for {url_str} using strategy: {extraction_source} (Content length: {len(content)})")
                return ExtractionResult(name=extraction_source, link=url_str, content=content)
            else:
                # Log if no content was extracted by *any* applicable strategy
                logger.warning(f"Extraction failed for URL {url_str}: No content could be extracted by any applicable strategy (Final Source Attempt: {extraction_source}).")
                return ExtractionResult(name=extraction_source, link=url_str, content=None, error=f"No content extracted by {extraction_source}")

        except (ValueError, ScrapingError, ConfigurationError) as e:
             logger.error(f"Scraping failed for {url}: {type(e).__name__}: {e}", exc_info=False)
             raise
        except Exception as e:
            logger.critical(f"Critical unexpected error during scrape() for {url}: {e}", exc_info=True)
            raise ScrapingError(f"An unexpected error occurred while scraping {url}: {e}") from e

    # --- scrape_many remains the same --- #
    async def scrape_many(
        self,
        urls: List[str],
        sequential: bool = True
    ) -> Dict[str, ExtractionResult]:
        if not urls:
            return {}
            
        results: Dict[str, ExtractionResult] = {}
        exceptions_count = 0
        total_urls = len(urls)

        if sequential:
            logger.info(f"Processing {total_urls} URLs sequentially...")
            for i, url in enumerate(urls):
                logger.debug(f"Scraping URL {i+1}/{total_urls} (sequential): {url}")
                try:
                    result: ExtractionResult = await self.scrape(url) # Call scrape without pdf args
                    results[url] = result
                except (ValueError, ScrapingError, ConfigurationError) as e:
                    exceptions_count += 1
                    logger.warning(f"Scraping failed for URL '{url}' (sequential): {type(e).__name__}: {e}", exc_info=False)
                except Exception as e:
                    exceptions_count += 1
                    logger.error(f"Unexpected error scraping URL '{url}' (sequential): {e}", exc_info=True)
        else:
            logger.info(f"Processing {total_urls} URLs concurrently...")
            # Pass self.scrape directly without extra args
            tasks = {url: asyncio.create_task(self.scrape(url)) for url in urls}

            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            url_list = list(tasks.keys())
            for i, result_or_exc in enumerate(task_results):
                url = url_list[i]
                if isinstance(result_or_exc, Exception):
                    exceptions_count += 1
                    e = result_or_exc
                    if isinstance(e, (ValueError, ScrapingError, ConfigurationError)):
                        logger.warning(f"Scraping failed for URL '{url}' (concurrent): {type(e).__name__}: {e}", exc_info=False)
                    else:
                        logger.error(f"Unexpected error scraping URL '{url}' (concurrent): {e}", exc_info=True)
                elif isinstance(result_or_exc, ExtractionResult):
                    results[url] = result_or_exc
                else:
                     exceptions_count += 1
                     logger.error(f"Unexpected return type from scrape for URL '{url}': {type(result_or_exc)}")

        if exceptions_count > 0:
            logger.warning(f"Completed scraping batch for {total_urls} URLs with {exceptions_count} failures.")
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