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
from urllib.parse import urlparse

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
        if not url or not urlparse(url).scheme in ['http', 'https']:
            raise ValueError(f"Invalid URL provided for scraping: {url}")

        parsed_url_obj = urlparse(url)
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
            elif url.lower().endswith('.pdf'):
                try:
                    logger.info(f"Dispatching to PDF handler for direct URL: {url}")
                    download_pdfs = self.settings.scraper_download_pdfs # Keep original file?
                    pdf_save_dir = self.settings.scraper_pdf_save_dir
                    
                    content = await pdf.handle_pdf_url(url, download_pdfs, pdf_save_dir, max_pdf_size_bytes)
                    extraction_source = "pdf"
                    logger.info(f"PDF handler successful for direct URL: {url}")
                except ScrapingError as e:
                    logger.error(f"PDF handling failed for direct URL {url}: {e}", exc_info=False)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in PDF handler for direct URL {url}: {e}", exc_info=True)
                    raise ScrapingError(f"PDF handling failed unexpectedly for direct URL {url}: {e}") from e

            # 3. General Web Crawling with Crawl4AI
            else:
                logger.info(f"Dispatching to Crawl4AI handler for general URL: {url}")
                crawl4ai_result: Optional[CrawlResult] = None
                try:
                    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                    # Use browser config from __init__ (already configured for downloads)
                    async with AsyncWebCrawler(config=self.browser_config) as crawler:
                        crawl4ai_result = await crawler.arun(url=url, config=run_config)

                    # --- Check for Downloaded PDF --- 
                    downloaded_pdf_path: Optional[str] = None
                    if hasattr(crawl4ai_result, 'downloaded_files') and crawl4ai_result.downloaded_files:
                        for file_path in crawl4ai_result.downloaded_files:
                             # Check if a downloaded file looks like a PDF
                             if isinstance(file_path, str) and file_path.lower().endswith('.pdf'):
                                 if os.path.exists(file_path):
                                     downloaded_pdf_path = file_path
                                     logger.info(f"Crawl4AI downloaded a PDF file: {downloaded_pdf_path}")
                                     break # Process the first valid PDF found
                                 else:
                                     logger.warning(f"Crawl4AI reported downloaded file, but path not found: {file_path}")

                    # --- Process Downloaded PDF OR Extracted Markdown --- 
                    if downloaded_pdf_path:
                        logger.info(f"Processing downloaded PDF: {downloaded_pdf_path}")
                        try:
                            # Process the local file
                            content = await pdf.handle_local_pdf_file(downloaded_pdf_path, max_pdf_size_bytes)
                            extraction_source = "crawl4ai_downloaded_pdf"
                            logger.info(f"Successfully processed downloaded PDF content from {downloaded_pdf_path}")
                            # Optionally delete the downloaded file if download_pdfs setting is False?
                            # if not self.settings.scraper_download_pdfs:
                            #    try: os.remove(downloaded_pdf_path) 
                            #    except OSError as rm_err: logger.warning(f"Could not delete downloaded PDF {downloaded_pdf_path}: {rm_err}")
                        except Exception as pdf_err:
                             logger.error(f"Failed to process downloaded PDF {downloaded_pdf_path}: {pdf_err}", exc_info=True)
                             # Fallback to markdown? Or raise error? Raising error for now.
                             raise ScrapingError(f"Failed processing downloaded PDF from {url}: {pdf_err}") from pdf_err
                    else:
                        # No PDF downloaded, process Markdown from Crawl4AI
                        logger.info(f"No PDF downloaded by Crawl4AI for {url}, attempting to use extracted Markdown.")
                        if hasattr(crawl4ai_result, 'success') and crawl4ai_result.success and hasattr(crawl4ai_result, 'markdown') and crawl4ai_result.markdown:
                            md_result = crawl4ai_result.markdown
                            if isinstance(md_result, MarkdownGenerationResult):
                                if hasattr(md_result, 'raw_markdown') and isinstance(md_result.raw_markdown, str):
                                    content = md_result.raw_markdown
                                    extraction_source = "crawl4ai_markdown_obj"
                            elif isinstance(md_result, str):
                                content = md_result
                                extraction_source = "crawl4ai_markdown_str"
                            else:
                                try: content = str(md_result)
                                except Exception: pass
                                if isinstance(content, str): extraction_source = "crawl4ai_markdown_converted"
                                else: content = None

                        if not isinstance(content, str) or not content.strip():
                            error_msg = getattr(crawl4ai_result, 'error', 'No valid markdown extracted and no PDF downloaded')
                            logger.error(f"Crawl4AI failed to provide usable content for {url}: {error_msg}")
                            raise ScrapingError(f"Crawl4AI failed for {url}: {error_msg}")
                        
                        logger.info(f"Using extracted Markdown ({extraction_source}) for {url}, Length: {len(content)}")
                        logger.debug(f"Crawl4AI Markdown Snippet for {url}:\n{content[:500]}...")

                except ScrapingError as e:
                    raise e # Re-raise known scraping errors
                except Exception as e:
                    logger.error(f"Error during Crawl4AI processing or download check for {url}: {e}", exc_info=True)
                    raise ScrapingError(f"Crawl4AI processing/download check failed for {url}: {e}") from e

            # --- Final Verification & Return --- #
            if content is None:
                logger.error(f"Scraping resulted in None content unexpectedly for {url} (source: {extraction_source})")
                raise ScrapingError(f"Extraction failed unexpectedly for {url} (content is None)")
            
            if not isinstance(content, str):
                 logger.critical(f"CRITICAL: Final content is not a string ({type(content)}) for {url}")
                 raise ScrapingError(f"Invalid final content type ({type(content)}) for {url}")

            if not content.strip():
                logger.warning(f"Extraction resulted in empty or whitespace-only content for {url} (source: {extraction_source}). Returning empty result.")
                return ExtractionResult(name=extraction_source, content="", raw_markdown_length=0)

            return ExtractionResult(
                 name=extraction_source,
                 content=content,
                 raw_markdown_length=len(content)
            )

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