# Scraper Refactoring and Integration Plan

**Goal:** Refactor the web scraping service (`app/services/scraping.py`) to primarily use Crawl4AI with Playwright for web content extraction, integrate `pymupdf4llm` for improved PDF processing, maintain Wikipedia and FastText quality filtering capabilities, and structure special handling logic modularly.

**Core Changes:**

1.  **Adopt Crawl4AI Markdown:** Rely on `crawl4ai`'s default Markdown generation (`result.markdown`) via the Playwright backend for general web pages, replacing custom HTML fetching and cleaning (`_fetch_raw_html`, `clean_html`).
2.  **Modular Special Scrapers:** Create a new directory `app/services/special_scrapers/` to house logic for handling specific URL types (e.g., Wikipedia, PDF).
3.  **Improved PDF Handling:** Use `pymupdf4llm` for PDF-to-Markdown conversion within the dedicated PDF special scraper module. Add a flag to control local saving of PDFs.
4.  **Integrate Quality Filtering:** Keep `fasttext`-based quality filtering but move it to a dedicated module (`app/services/quality_filter.py`).
5.  **Streamline `WebScraper`:** Simplify the main `WebScraper` class to act primarily as a dispatcher based on URL type and orchestrator for the general Crawl4AI process and quality filtering.

---

## Detailed Steps:

**Phase 1: Setup and Module Creation**

1.  **Create Directory Structure:**
    *   Create `app/services/special_scrapers/`
    *   Create `app/services/quality_filter.py`
2.  **Create Initial Files:**
    *   Create `app/services/special_scrapers/__init__.py` (can be empty)
    *   Create `app/services/special_scrapers/wikipedia.py`
    *   Create `app/services/special_scrapers/pdf.py`

**Phase 2: Migrate Existing Logic to Modules**

1.  **Move Wikipedia Logic:**
    *   **Action:** Move the `get_wikipedia_content` function and its dependencies (`wikipediaapi`, relevant imports) from `app/services/scraping.py` to `app/services/special_scrapers/wikipedia.py`.
    *   **Ensure:** Imports within `wikipedia.py` are correct. Add necessary error handling (e.g., `ScrapingError` defined or imported).
2.  **Move and Refactor PDF Logic:**
    *   **Action:** Move PDF-related logic (`extract_from_pdf_url`, `_parse_pdf_bytes`, `httpx` dependency, relevant imports) from `app/services/scraping.py` to `app/services/special_scrapers/pdf.py`.
    *   **Refactor:**
        *   Rename `extract_from_pdf_url` to `handle_pdf_url`.
        *   Modify `handle_pdf_url` to accept `url: str`, `download_pdfs: bool`, `save_dir: str`.
        *   Use `httpx` (or `requests`) to download PDF bytes *if* `download_pdfs` is `True`, saving to `save_dir`. Handle potential download errors.
        *   Always attempt to download PDF bytes into memory for processing, regardless of the `download_pdfs` flag.
        *   Replace the internal call to `_parse_pdf_bytes` with a call to `pymupdf4llm.to_markdown()` using the in-memory bytes. Handle `pymupdf4llm` errors.
        *   Remove the old `_parse_pdf_bytes` function.
    *   **Dependencies:** Add `pymupdf4llm` to requirements and import it in `pdf.py`. Remove `pypdfium2`.
3.  **Move Quality Filtering Logic:**
    *   **Action:** Move FastText-related functions (`_load_fasttext_model`, `_fasttext_model`, `_fasttext_load_lock`, `predict_educational_value`, `clean_markdown_links`, `filter_quality_content`, `replace_newlines`, `score_dict`) and their dependencies (`fasttext`, `re`, `asyncio`, `hf_hub_download`, `logging`, `ConfigurationError`) from `app/services/scraping.py` to `app/services/quality_filter.py`.
    *   **Ensure:** Model loading (`_load_fasttext_model`) is robust and handles exceptions appropriately.

**Phase 3: Refactor `WebScraper` (`app/services/scraping.py`)**

1.  **Update Imports:** Add imports for the new modules (e.g., `from .special_scrapers import wikipedia, pdf`, `from . import quality_filter`). Remove unused imports (`bs4`, `wikipediaapi`, `pypdfium2`, strategy-related, etc.).
2.  **Simplify `__init__`:**
    *   Remove parameters and attributes related to `strategies`, `llm_instruction`, `llm_api_key`, `llm_model`, `user_query`.
    *   Remove calls to `_create_extraction_configs`.
    *   Keep `debug` flag.
    *   Ensure `BrowserConfig` defaults to or is set to use Playwright (`backend='playwright'`). Explicitly set `verbose=False` in `BrowserConfig` to reduce noise.
3.  **Rewrite `scrape` Method:**
    *   **Signature:** `async def scrape(self, url: str, download_pdfs: bool = False, pdf_save_dir: str = "downloaded_pdfs") -> ExtractionResult:` (or similar return type).
    *   **URL Dispatching Logic:**
        ```python
        parsed_url = urlparse(url)
        content = None
        extraction_source = "unknown" # To track origin for result

        if "wikipedia.org" in parsed_url.netloc:
            try:
                logger.info(f"Dispatching to Wikipedia handler for: {url}")
                content = await wikipedia.get_wikipedia_content(url)
                extraction_source = "wikipedia"
            except Exception as e:
                # Handle/log Wikipedia-specific errors, maybe raise ScrapingError
                raise ScrapingError(f"Wikipedia extraction failed for {url}: {e}") from e
        elif url.lower().endswith('.pdf'):
            try:
                logger.info(f"Dispatching to PDF handler for: {url}")
                # handle_pdf_url should return the extracted markdown text
                content = await pdf.handle_pdf_url(url, download_pdfs, pdf_save_dir)
                extraction_source = "pdf"
            except Exception as e:
                # Handle/log PDF-specific errors, maybe raise ScrapingError
                raise ScrapingError(f"PDF handling failed for {url}: {e}") from e
        else:
            # General Web Crawling
            logger.info(f"Dispatching to Crawl4AI handler for: {url}")
            try:
                # Configure crawler within the method scope
                # Use a simple run config, relying on default markdown generation
                run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS) 
                # Define browser config here or ensure self.browser_config is Playwright
                browser_cfg = BrowserConfig(headless=True, verbose=False, backend='playwright') 

                async with AsyncWebCrawler(config=browser_cfg) as crawler:
                    # Use arun for single URL
                    result = await crawler.arun(url=url, config=run_config) 

                if result.success and result.markdown:
                    content = result.markdown
                    extraction_source = "crawl4ai_markdown"
                    # Log length, not content
                    logger.info(f"Crawl4AI success for {url}, Markdown length: {len(content)}")
                else:
                    error_msg = getattr(result, 'error', 'Unknown crawl4ai error')
                    logger.warning(f"Crawl4AI failed for {url}: {error_msg}")
                    # Decide if this is a critical error or should return empty/None
                    raise ScrapingError(f"Crawl4AI failed for {url}: {error_msg}")

            except Exception as e:
                logger.error(f"General web scraping error for {url}: {e}", exc_info=True)
                raise ScrapingError(f"General web scraping failed for {url}: {e}") from e

        # --- Quality Filtering ---
        final_content = None
        if content:
            try:
                logger.info(f"Applying quality filter for content from {url} ({extraction_source})")
                # Assuming filter_quality_content takes string and returns string
                final_content = await quality_filter.filter_quality_content(content, self.min_quality_score) # Need self.min_quality_score from init
                logger.info(f"Quality filtering complete for {url}. Original length: {len(content)}, Filtered length: {len(final_content)}")
            except Exception as e:
                 logger.error(f"Quality filtering failed for {url}: {e}. Using unfiltered content.", exc_info=True)
                 final_content = content # Fallback to unfiltered content
        else:
            logger.warning(f"No content extracted for {url} ({extraction_source}) prior to quality filtering.")
            # Ensure we return an appropriate result for failure cases
            # Maybe raise error earlier if content is None after extraction attempt

        # --- Return Result ---
        if final_content is not None:
            # Adapt ExtractionResult or return a simple dict
             return ExtractionResult(
                 name=extraction_source, # Use the source name
                 content=final_content,
                 raw_markdown_length=len(final_content) # Or maybe original content length?
             )
        else:
             # Handle case where no content could be extracted or filtered
             logger.error(f"Returning empty result for {url} as no content was finalized.")
             # Return an empty result or raise an error depending on desired behavior
             return ExtractionResult(name=extraction_source, content=None) # Indicate failure
        ```
    *   Add `min_quality_score` parameter to `__init__` and store it.
4.  **Refactor `scrape_many`:** Update it to call the new `scrape` method signature, passing `download_pdfs` and `pdf_save_dir` if needed (perhaps add these as optional args to `scrape_many`).
5.  **Remove Unused Code:** Delete `_fetch_raw_html`, `clean_html`, `replace_*` functions, `StrategyFactory`, `ExtractionConfig` (if no longer used), old `ExtractionResult` fields if simplified, strategy-related imports/logic.

**Phase 4: Dependencies and Testing**

1.  **Update `requirements.txt`:** Add `pymupdf4llm`, `wikipedia-api`. Ensure `crawl4ai[all]` (or specific backend needed), `fasttext`, `huggingface_hub`, `python-dotenv` are present. Remove `pypdfium2`. Review if `requests` is still needed besides Serper calls.
2.  **Install Dependencies:** Run `pip install -r requirements.txt` and `playwright install`.
3.  **Testing:**
    *   Test Wikipedia URLs.
    *   Test PDF URLs (with `download_pdfs=True` and `False`). Check extracted Markdown quality.
    *   Test various general web URLs (dynamic JS sites, simple HTML sites). Check Markdown quality.
    *   Verify quality filtering is applied.
    *   Test `scrape_many`.

---

This plan provides a structured approach. We will tackle it phase by phase, updating this guide as we complete steps or encounter issues.