"""
Service for scraping web content using Crawl4AI, specialized handlers, and quality filtering.
"""

import asyncio
# import os # Can be removed if load_dotenv handles API keys sufficiently elsewhere
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional # Removed Tuple, Union unless needed by scrape_many/scrape
from urllib.parse import urlparse, urljoin

# External Dependencies
import httpx # Potentially remove if only used in pdf.py
from dotenv import load_dotenv # Keep if OPENROUTER_API_KEY or other envs needed here

# Crawl4AI Imports
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MarkdownGenerationResult # Import the specific result type
)

# Import configuration
from ..core.config import AppSettings

# Import custom exceptions
from ..core.exceptions import ScrapingError, ConfigurationError

# Import local modules
from .special_scrapers import wikipedia, pdf
# from . import quality_filter # Removed quality filter import

logger = logging.getLogger(__name__)

# Ensure environment variables are loaded if needed (e.g., for API keys used indirectly)
load_dotenv()

# --- Dataclass for Result --- #
@dataclass
class ExtractionResult:
    """Holds the results of a scraping operation."""
    name: str # Source: 'wikipedia', 'pdf', 'crawl4ai_markdown'
    content: Optional[str] = None
    raw_markdown_length: Optional[int] = None # Now represents raw extracted length

# --- WebScraper Class --- #
class WebScraper:
    """Coordinates web scraping using specialized handlers or Crawl4AI."""

    def __init__(
        self,
        settings: AppSettings, # Add AppSettings dependency
        debug: bool = False, # Keep debug separate for now
        # min_quality_score: float = 0.2, # Removed min_quality_score
        browser_config: Optional[BrowserConfig] = None
    ):
        """Initializes the WebScraper.

        Args:
            settings: The application settings instance.
            debug: Enable debug logging.
            browser_config: Optional custom BrowserConfig for Crawl4AI.
        """
        self.settings = settings
        self.debug = debug
        # self.min_quality_score = min_quality_score # Removed

        self.browser_config = browser_config or BrowserConfig(
            browser_type='chromium',
            headless=True,
            verbose=self.debug,
        )

        logger.info(
            f"WebScraper initialized. Debug: {self.debug}, "
            f"Browser Type: {self.browser_config.browser_type}"
            # Removed quality score from log
        )

    async def scrape(self, url: str) -> ExtractionResult: # Removed pdf args
        """Scrapes content from a URL, dispatching to specific handlers or using Crawl4AI.
           Uses settings for PDF handling (download, path, size limit).
        """
        if not url or not urlparse(url).scheme in ['http', 'https']:
            raise ValueError(f"Invalid URL provided for scraping: {url}")

        # Get PDF settings from AppSettings
        download_pdfs = self.settings.scraper_download_pdfs
        pdf_save_dir = self.settings.scraper_pdf_save_dir
        max_pdf_size_bytes = self.settings.scraper_max_pdf_size_mb * 1024 * 1024

        parsed_url_obj = urlparse(url)
        content: Optional[str] = None
        extraction_source = "unknown" # To track origin for result

        try:
            # --- URL Dispatching --- #
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

            elif url.lower().endswith('.pdf'):
                try:
                    logger.info(f"Dispatching to PDF handler for: {url}")
                    content = await pdf.handle_pdf_url(url, download_pdfs, pdf_save_dir, max_pdf_size_bytes)
                    extraction_source = "pdf"
                    logger.info(f"PDF handler successful for: {url}")
                except ScrapingError as e:
                    logger.error(f"PDF handling failed for {url}: {e}", exc_info=False)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in PDF handler for {url}: {e}", exc_info=True)
                    raise ScrapingError(f"PDF handling failed unexpectedly for {url}: {e}") from e

            else:
                # --- General Web Crawling with Crawl4AI --- #
                logger.info(f"Dispatching to Crawl4AI handler for: {url}")
                initial_content_str: Optional[str] = None
                crawl4ai_source = "unknown"
                pdf_link_to_process: Optional[str] = None
                crawl4ai_result = None # Initialize result object

                try:
                    # --- Run Crawl4AI ---
                    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                    browser_cfg = self.browser_config
                    async with AsyncWebCrawler(config=browser_cfg) as crawler:
                        crawl4ai_result = await crawler.arun(url=url, config=run_config)

                    # --- Process Crawl4AI Result (Get initial Markdown string) ---
                    if hasattr(crawl4ai_result, 'success') and crawl4ai_result.success and hasattr(crawl4ai_result, 'markdown') and crawl4ai_result.markdown:
                        md_result = crawl4ai_result.markdown
                        if isinstance(md_result, MarkdownGenerationResult):
                            if hasattr(md_result, 'raw_markdown') and isinstance(md_result.raw_markdown, str):
                                initial_content_str = md_result.raw_markdown
                                crawl4ai_source = "crawl4ai_markdown_obj"
                        elif isinstance(md_result, str):
                            initial_content_str = md_result
                            crawl4ai_source = "crawl4ai_markdown_str"
                        else: # Fallback using str()
                            try: initial_content_str = str(md_result)
                            except Exception: pass
                            if isinstance(initial_content_str, str): crawl4ai_source = "crawl4ai_markdown_converted"
                            else: initial_content_str = None

                    # Check if initial markdown extraction succeeded
                    if not isinstance(initial_content_str, str) or not initial_content_str.strip():
                        error_msg = getattr(crawl4ai_result, 'error', 'Unknown error or no valid markdown extracted')
                        raise ScrapingError(f"Crawl4AI failed to extract valid initial markdown for {url}: {error_msg}")

                    logger.info(f"Crawl4AI initial success for {url} ({crawl4ai_source}), Length: {len(initial_content_str)}")
                    logger.debug(f"Initial Markdown Snippet for {url}:\n{initial_content_str[:500]}...") # Shorter snippet

                    # --- Find PDF link using crawl4ai_result.links ---
                    if hasattr(crawl4ai_result, 'links') and isinstance(crawl4ai_result.links, dict):
                         all_links = crawl4ai_result.links.get("internal", []) + crawl4ai_result.links.get("external", [])
                         for link_info in all_links:
                             href = link_info.get("href")
                             if href and isinstance(href, str) and "/pdf/" in href.lower():
                                 absolute_pdf_url = urljoin(url, href)
                                 if absolute_pdf_url.lower().endswith(".pdf") or "/pdf/" in absolute_pdf_url.lower():
                                     pdf_link_to_process = absolute_pdf_url
                                     logger.info(f"Found potential PDF link via result.links: {pdf_link_to_process}")
                                     break

                except ScrapingError as e:
                    raise e # Re-raise known scraping errors
                except Exception as e:
                    logger.error(f"Error during initial Crawl4AI processing or link finding for {url}: {e}", exc_info=True)
                    raise ScrapingError(f"Initial Crawl4AI processing/link finding failed for {url}: {e}") from e

                # --- Process Linked PDF if Found --- #
                final_content_str = initial_content_str # Start with initial markdown
                final_source = crawl4ai_source

                if pdf_link_to_process:
                    logger.info(f"Attempting to fetch content from linked PDF: {pdf_link_to_process}")
                    try:
                        # Use settings for PDF handling here too
                        pdf_content = await pdf.handle_pdf_url(pdf_link_to_process, download_pdfs, pdf_save_dir, max_pdf_size_bytes)
                        if isinstance(pdf_content, str) and pdf_content.strip():
                            logger.info(f"Successfully extracted content from linked PDF: {pdf_link_to_process}. Appending.")
                            # Combine content
                            final_content_str = f"{initial_content_str}\n\n<hr/>\n\n**Appended PDF Content ({pdf_link_to_process}):**\n\n{pdf_content}"
                            final_source += "+appended_pdf"
                        else:
                            logger.warning(f"Extraction from linked PDF {pdf_link_to_process} yielded empty or non-string content. Not appending.")
                    except Exception as pdf_err:
                        logger.warning(f"Failed to process linked PDF {pdf_link_to_process}: {pdf_err}", exc_info=False)
                else:
                     logger.info(f"No suitable PDF link found via result.links for {url}.")

                # --- Assign final verified string content --- #
                content = final_content_str
                extraction_source = final_source

            # --- Final Verification & Return (Applies to all paths) --- #
            if content is None: # Should only be reachable if Wiki/DirectPDF failed
                logger.error(f"Content is None before returning for {url} (source: {extraction_source})")
                raise ScrapingError(f"Extraction failed unexpectedly for {url} (content is None)")
            if not isinstance(content, str): # Final safeguard
                 logger.critical(f"CRITICAL: Content is not a string ({type(content)}) just before returning for {url}")
                 raise ScrapingError(f"Invalid content type ({type(content)}) before return for {url}")

            if not content.strip(): # Handle empty strings from any source
                logger.warning(f"Extraction resulted in empty or whitespace-only content for {url} (source: {extraction_source}). Returning empty result.")
                return ExtractionResult(name=extraction_source, content="", raw_markdown_length=0)

            # Return the final result (content is guaranteed non-empty string)
            return ExtractionResult(
                 name=extraction_source,
                 content=content,
                 raw_markdown_length=len(content)
            )

        # Catch errors raised during dispatching/filtering
        except (ValueError, ScrapingError, ConfigurationError) as e:
             logger.error(f"Scraping failed for {url}: {type(e).__name__}: {e}", exc_info=False)
             raise
        except Exception as e:
            logger.critical(f"Critical unexpected error during scrape for {url}: {e}", exc_info=True)
            raise ScrapingError(f"An unexpected error occurred while scraping {url}: {e}") from e


    async def scrape_many(
        self,
        urls: List[str],
        # Removed pdf args, will be taken from self.settings
        sequential: bool = True # Default to sequential processing
    ) -> Dict[str, ExtractionResult]:
        """Scrapes multiple URLs, handling errors individually.
        
        Args:
            urls: List of URLs to scrape.
            sequential: If True, scrape URLs one by one.
            
        Returns:
            A dictionary mapping successfully scraped URLs to their ExtractionResult.
            Failed URLs are logged but not included in the returned dict.
        """
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