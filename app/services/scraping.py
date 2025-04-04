"""
Service for scraping web content using Crawl4AI and various strategies.
Consolidates functionality previously spread across multiple files.
"""

import asyncio
import os
import re
import json
import logging # Added logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

# External Dependencies (Ensure these are in requirements.txt)
import fasttext
import wikipediaapi
import httpx # Using httpx for network calls
from bs4 import BeautifulSoup, SoupStrainer # Added SoupStrainer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.extraction_strategy import (
    ExtractionStrategy,
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    NoExtractionStrategy,
    CosineStrategy,
)

# Import custom exceptions
from ..core.exceptions import (
    ScrapingError, ConfigurationError, LLMCommunicationError # Added LLM specific for strategy
)

logger = logging.getLogger(__name__)

# --- Dependencies moved from extraction_result.py ---
@dataclass
class ExtractionResult:
    """Holds the results of an extraction operation (now primarily successful content)"""
    name: str
    content: Optional[str] = None
    raw_markdown_length: int = field(default=0)
    citations_markdown_length: int = field(default=0)

# --- Dependencies moved from basic_web_scraper.py ---
@dataclass
class ExtractionConfig:
    """Configuration for extraction strategies"""
    name: str
    strategy: ExtractionStrategy

# --- Dependencies moved from strategy_factory.py ---
class StrategyFactory:
    """Factory for creating extraction strategies"""
    @staticmethod
    def create_llm_strategy(
        input_format: str = "markdown",
        instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMExtractionStrategy:
        load_dotenv() # Ensure env vars are loaded if key not passed
        used_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        used_model = model or "openrouter/google/gemini-flash-1.5" # Default if not provided

        if not used_api_key:
            raise ConfigurationError("OpenRouter API key is required for LLM extraction strategy but not found.")

        return LLMExtractionStrategy(
            input_format=input_format,
            provider=used_model,
            api_token=used_api_key,
            instruction=instruction
        )

    @staticmethod
    def create_css_strategy() -> JsonCssExtractionStrategy:
        schema = {
            "baseSelector": ".product",
            "fields": [
                {"name": "title", "selector": "h1.product-title", "type": "text"},
                {"name": "price", "selector": ".price", "type": "text"},
                {"name": "description", "selector": ".description", "type": "text"},
            ],
        }
        return JsonCssExtractionStrategy(schema=schema)

    @staticmethod
    def create_xpath_strategy() -> JsonXPathExtractionStrategy:
        schema = {
            "baseSelector": "//div[@class='product']",
            "fields": [
                {"name": "title", "selector": ".//h1[@class='product-title']/text()", "type": "text"},
                {"name": "price", "selector": ".//span[@class='price']/text()", "type": "text"},
                {"name": "description", "selector": ".//div[@class='description']/text()", "type": "text"},
            ],
        }
        return JsonXPathExtractionStrategy(schema=schema)

    @staticmethod
    def create_no_extraction_strategy() -> NoExtractionStrategy:
        return NoExtractionStrategy()

    @staticmethod
    def create_cosine_strategy(
        semantic_filter: Optional[str] = None,
        word_count_threshold: int = 10,
        max_dist: float = 0.2,
        sim_threshold: float = 0.3,
        debug: bool = False
    ) -> CosineStrategy:
        return CosineStrategy(
            semantic_filter=semantic_filter,
            word_count_threshold=word_count_threshold,
            max_dist=max_dist,
            sim_threshold=sim_threshold,
            verbose=debug
        )

# --- Dependencies moved from utils.py ---

# Global variable to hold the loaded model, initialized to None
_fasttext_model: Optional[fasttext.FastText._FastText] = None
_fasttext_load_lock = asyncio.Lock()

async def _load_fasttext_model():
    """Loads the FastText model asynchronously and thread-safely if not already loaded."""
    global _fasttext_model
    if _fasttext_model is None:
        async with _fasttext_load_lock:
            # Double-check after acquiring the lock
            if _fasttext_model is None:
                logger.info("Loading FastText quality model...")
                try:
                    # Running synchronous hf_hub_download in executor
                    loop = asyncio.get_running_loop()
                    model_path = await loop.run_in_executor(
                        None, # Use default executor
                        hf_hub_download,
                        "kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2",
                        "model.bin"
                    )
                    # fasttext.load_model is also sync
                    _fasttext_model = await loop.run_in_executor(None, fasttext.load_model, model_path)
                    logger.info("FastText quality model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load FastText model: {e}", exc_info=True)
                    # Raise a configuration error as quality filtering cannot proceed
                    raise ConfigurationError(f"Failed to load FastText quality model: {e}") from e

def replace_newlines(text: str) -> str:
    """Replace multiple newlines with a single space."""
    return re.sub(r"\n+", " ", text)

score_dict = {
    '__label__Low': 0,
    '__label__Mid': 1,
    '__label__High': 2
}

async def predict_educational_value(text_list: List[str]) -> List[float]:
    """
    Predict educational value scores for a list of texts asynchronously.
    Returns a list of scores between 0 and 2.
    Loads the model lazily and thread-safely on first call.

    Raises:
        ConfigurationError: If the FastText model failed to load previously.
        RuntimeError: If prediction fails unexpectedly.
    """
    # Ensure the model is loaded
    await _load_fasttext_model()

    # Check if model loading failed (should have raised ConfigurationError in _load_fasttext_model)
    if _fasttext_model is None:
        # This state should ideally not be reached if _load_fasttext_model raises correctly
        raise ConfigurationError("FastText model is not available for quality prediction.")

    try:
        # Run prediction in executor as predict method is synchronous
        loop = asyncio.get_running_loop()
        processed_texts = [replace_newlines(text) for text in text_list]
        # The predict method itself seems GIL-bound but might be slow, run in executor
        pred_labels, pred_scores = await loop.run_in_executor(
            None, _fasttext_model.predict, processed_texts, -1
        )

        score_list = []
        for labels, scores in zip(pred_labels, pred_scores):
            score = 0.0
            for label, score_val in zip(labels, scores):
                score += score_dict.get(label, 0) * score_val
            score_list.append(float(score))
        return score_list
    except Exception as e:
        logger.error(f"Error during FastText prediction: {e}", exc_info=True)
        raise RuntimeError(f"Failed to predict educational value: {e}") from e

async def clean_markdown_links(text: str, min_quality_score: float = 0.2) -> Tuple[str, float]:
    """
    Clean markdown links and filter low-quality content asynchronously.
    Returns tuple of (cleaned_text, quality_score)

    Raises:
        ConfigurationError: If FastText model loading failed.
        RuntimeError: If prediction fails.
    """
    # Split by double newlines to preserve paragraph structure
    paragraphs = text.split('\n\n')

    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Preserve code blocks by checking if paragraph contains ``` tags
        if '```' in paragraph:
            cleaned_paragraphs.append(paragraph)
            continue

        lines = paragraph.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Keep headers regardless of length
            if re.match(r'^#{1,6}\s+', line):
                filtered_lines.append(line)
                continue

            # Skip common UI/navigation elements
            if re.match(r'^(Share|Trade|More|Buy|Sell|Download|Menu|Home|Back|Next|Previous|\d+\s*(BTC|USD|EUR|GBP)|\w{3}-\w{1,3}|Currency:.*|You (Buy|Spend|Receive)|≈|\d+\.\d+)', line, re.IGNORECASE):
                continue

            # Count words before removing markdown
            word_count = len(re.sub(r'\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>', '', line).split())

            # Increase minimum word threshold to 12
            if word_count < 12:
                # Check if line only contains markdown patterns or appears to be a currency/trading related line
                cleaned_line = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)|\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>|\d+(\.\d+)?%?|\$\d+(\.\d+)?', '', line).strip()
                if not cleaned_line or len(cleaned_line.split()) < 8:  # If nothing substantial remains, skip this line
                    continue

            filtered_lines.append(line)

        # Only add paragraph if it has any lines left
        if filtered_lines:
            cleaned_paragraphs.append('\n'.join(filtered_lines))

    # Rejoin with double newlines
    cleaned_text = '\n\n'.join(cleaned_paragraphs)

    # Get quality score asynchronously
    quality_scores = await predict_educational_value([cleaned_text])
    quality_score = quality_scores[0]

    return cleaned_text, quality_score

async def filter_quality_content(text: str, min_quality_score: float = 0.2) -> str:
    """
    Filter content based on quality and returns concatenated quality content asynchronously.

    Raises:
        ConfigurationError: If FastText model loading failed.
        RuntimeError: If prediction fails.
    """
    paragraphs = text.split('\n\n')
    quality_content = []
    tasks = []

    # Create tasks for cleaning/scoring each paragraph
    for paragraph in paragraphs:
        if paragraph.strip():
            tasks.append(clean_markdown_links(paragraph, min_quality_score))
        else:
            # Handle empty paragraphs if needed, maybe just skip
            pass

    # Run tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for result in results:
        if isinstance(result, Exception):
            # Log the error and decide whether to propagate or skip
            logger.error(f"Error cleaning/scoring paragraph: {result}", exc_info=False) # Don't need full traceback here usually
            # Re-raise the first critical error encountered (e.g. ConfigurationError)
            if isinstance(result, (ConfigurationError, RuntimeError)):
                raise result # Propagate critical errors
            # Otherwise, we might just skip this paragraph for quality filtering
            continue
        elif isinstance(result, tuple):
            cleaned_text, quality_score = result
            if cleaned_text and quality_score >= min_quality_score:
                quality_content.append(cleaned_text)

    logger.debug(f"Found {len(quality_content)} quality paragraphs out of {len(paragraphs)} total")

    if quality_content:
        return "\n\n".join(quality_content)
    logger.warning("No paragraphs met the quality threshold. Returning original text.")
    return text # Return original text if no quality content found or errors occurred in all paragraphs

async def get_wikipedia_content(url: str, lang: str = 'en') -> str:
    """
    Extract content from a Wikipedia URL asynchronously.

    Args:
        url: The Wikipedia URL.
        lang: Language code for Wikipedia (default: 'en').

    Returns:
        The main text content of the Wikipedia page.

    Raises:
        ValueError: If the URL is not a valid Wikipedia URL or page title cannot be extracted.
        ScrapingError: If the page doesn't exist or there's an error fetching/parsing content.
    """
    parsed_url = urlparse(url)
    if not parsed_url.netloc.endswith("wikipedia.org"):
        raise ValueError(f"URL '{url}' is not a valid Wikipedia URL.")

    # Extract title from the path
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2 or path_parts[0] != 'wiki':
        raise ValueError(f"Could not extract page title from Wikipedia URL path: {parsed_url.path}")
    page_title = path_parts[-1]

    try:
        wiki_api = wikipediaapi.Wikipedia(
            user_agent='DeepResearchAgent/0.1 (contact@example.com)', # Replace with actual contact info
            language=lang,
            extract_format=wikipediaapi.ExtractFormat.WIKI # Or .HTML
        )

        # Run sync wikipediaapi calls in executor
        loop = asyncio.get_running_loop()
        page = await loop.run_in_executor(None, wiki_api.page, page_title)

        if not page.exists():
            raise ScrapingError(f"Wikipedia page '{page_title}' ('{url}') does not exist.")

        # Extract text content
        content = page.text # Access the summary or full text depending on API setup
        if not content:
            logger.warning(f"Extracted empty content from Wikipedia page: {url}")
            # Decide if empty content is an error or just an empty page
            # Raising error for now, as usually expect some text
            raise ScrapingError(f"Failed to extract text content from Wikipedia page: {url}")

        logger.info(f"Successfully extracted content from Wikipedia URL: {url}")
        return content

    except wikipediaapi.exceptions.WikipediaException as e:
        logger.error(f"Wikipedia API error for URL '{url}': {e}", exc_info=True)
        raise ScrapingError(f"Wikipedia API error for '{url}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error fetching Wikipedia content for URL '{url}': {e}", exc_info=True)
        raise ScrapingError(f"Unexpected error fetching Wikipedia content for '{url}': {e}") from e

# Patterns for cleaning HTML
SCRIPT_PATTERN = r"<[ ]*script.*?\\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\\/svg>)"
IFRAME_PATTERN = r"<[ ]*iframe.*?\\/[ ]*iframe[ ]*>"
NOSCRIPT_PATTERN = r"<[ ]*noscript.*?\\/[ ]*noscript[ ]*>"
HEADER_PATTERN = r"<[ ]*header.*?\\/[ ]*header[ ]*>"
FOOTER_PATTERN = r"<[ ]*footer.*?\\/[ ]*footer[ ]*>"
NAV_PATTERN = r"<[ ]*nav.*?\\/[ ]*nav[ ]*>"
FORM_PATTERN = r"<[ ]*form.*?\\/[ ]*form[ ]*>"


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    """Clean HTML content by removing various elements."""
    patterns = [
        SCRIPT_PATTERN,
        STYLE_PATTERN,
        META_PATTERN,
        COMMENT_PATTERN,
        LINK_PATTERN,
        IFRAME_PATTERN,
        NOSCRIPT_PATTERN,
        HEADER_PATTERN,
        FOOTER_PATTERN,
        NAV_PATTERN,
        FORM_PATTERN
    ]

    for pattern in patterns:
        html = re.sub(pattern, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)

    # Remove empty lines and excessive whitespace
    html = re.sub(r'\n\s*\n', '\n', html)
    html = re.sub(r'\s+', ' ', html)

    return html.strip()


# Helper function to detect academic paper URLs
def is_academic_url(url: str) -> bool:
    """Check if URL is likely from an academic source or repository."""
    academic_domains = [
        'arxiv.org', 'scholar.google.com', 'researchgate.net', 'ncbi.nlm.nih.gov', 
        'pubmed.ncbi.nlm.nih.gov', 'sciencedirect.com', 'nature.com', 'science.org',
        'ieee.org', 'acm.org', 'springer.com', 'wiley.com', 'tandfonline.com', 
        'jstor.org', 'ssrn.com', 'academia.edu', 'biorxiv.org', 'medrxiv.org'
    ]
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    return any(academic_domain in domain for academic_domain in academic_domains)

# Placeholder for PDF extraction refactor - requires installing pypdfium2 or similar
async def extract_from_pdf_url(url: str, debug: bool = False) -> str:
    """Placeholder: Extracts text content from a PDF URL asynchronously."""
    logger.warning(f"PDF extraction from URL '{url}' is not fully implemented with robust error handling yet.")
    # TODO: Implement using httpx to fetch, pypdfium2 (in executor) to parse
    # Catch httpx errors, PDF parsing errors, raise ScrapingError
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client: # Increased timeout for download
            logger.debug(f"Fetching PDF from URL: {url}")
            response = await client.get(url)
            response.raise_for_status() # Check for HTTP errors
            pdf_bytes = await response.aread()
            logger.debug(f"Successfully downloaded PDF bytes from {url}")

        # Run pypdfium2 parsing in executor
        loop = asyncio.get_running_loop()
        logger.debug(f"Parsing PDF content for {url}")
        text = await loop.run_in_executor(None, _parse_pdf_bytes, pdf_bytes)
        logger.info(f"Successfully extracted text from PDF: {url}")
        return text

    except httpx.TimeoutException as e:
        logger.error(f"Timeout fetching PDF from {url}: {e}")
        raise ScrapingError(f"Timeout fetching PDF from {url}: {e}") from e
    except httpx.HTTPStatusError as e:
        # Log only status code and brief message, not full HTML response
        status_code = e.response.status_code
        error_msg = f"HTTP error {status_code} fetching PDF"
        logger.error(f"{error_msg} from {url} - response too large to log")
        raise ScrapingError(f"{error_msg} from {url}") from e
    except httpx.RequestError as e:
        logger.error(f"Network error fetching PDF from {url}: {e}")
        raise ScrapingError(f"Network error fetching PDF from {url}: {e}") from e
    except ImportError:
         logger.error("pypdfium2 library not found. Cannot parse PDF.")
         raise ConfigurationError("PDF parsing library (pypdfium2) is not installed.")
    except Exception as e: # Catch pypdfium2 specific errors (e.g., PdfError) if possible
        # Need to import PdfError from pypdfium2 if installed
        # except pdfium.PdfError as e:
        #    logger.error(f"Failed to parse PDF content from {url}: {e}")
        #    raise ScrapingError(f"Failed to parse PDF content from {url}: {e}") from e
        logger.error(f"Unexpected error extracting PDF from {url}: {e}", exc_info=True)
        raise ScrapingError(f"Unexpected error extracting PDF from {url}: {e}") from e

def _parse_pdf_bytes(pdf_bytes: bytes) -> str:
    """Synchronous function to parse PDF bytes using pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError("pypdfium2 is required for PDF parsing but not installed.")

    text = ""
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            textpage = page.get_textpage()
            text += textpage.get_text_range() + "\n"
            # Ensure resources are released
            textpage.close()
            page.close()
        pdf.close()
        return text
    except pdfium.PdfiumError as e:
        # Catch specific pypdfium2 errors
        raise RuntimeError(f"pypdfium2 failed to parse PDF: {e}") from e
    except Exception as e:
        # Catch other unexpected errors during parsing
        raise RuntimeError(f"Unexpected error during PDF parsing: {e}") from e

# --- WebScraper Class --- #

class WebScraper:
    """Handles scraping web pages using various strategies via Crawl4AI."""
    def __init__(
        self,
        strategies: List[str] = ['no_extraction'],
        llm_instruction: Optional[str] = None, # Default defined in factory
        llm_api_key: Optional[str] = None, # Allow passing key
        llm_model: Optional[str] = None, # Allow passing model
        user_query: Optional[str] = None,
        debug: bool = False,
        filter_content: bool = False,
        min_quality_score: float = 0.2,
        browser_config: Optional[BrowserConfig] = None
    ):
        self.debug = debug
        self.filter_content = filter_content
        self.min_quality_score = min_quality_score
        self.user_query = user_query
        self.llm_instruction = llm_instruction
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model

        try:
            self.extraction_configs = self._create_extraction_configs(strategies)
        except ConfigurationError as e:
             logger.critical(f"Failed to initialize WebScraper due to strategy configuration error: {e}")
             # Critical failure if a required strategy cannot be configured
             raise # Re-raise ConfigurationError

        # Configure Browser - Reverted to simpler config like oldscraper.py
        # Disable verbose logging from the browser to reduce console noise
        self.browser_config = browser_config or BrowserConfig(headless=True, verbose=False)

        # Configure Crawler Run
        self.run_config = self._create_crawler_config()

        if not self.extraction_configs:
            # This case should be prevented by the check in __init__ raising ConfigurationError
            # If reached, it indicates a logic error or non-critical strategy failure in _create_extraction_configs
            logger.warning("WebScraper initialized with no valid extraction strategies.")
            # Decide if this is acceptable or should raise an error
            # raise ConfigurationError("WebScraper requires at least one valid extraction strategy.")

    def _create_extraction_configs(self, strategy_names: List[str]) -> List[ExtractionConfig]:
        # (Keep existing logic, ensuring it raises ConfigurationError if needed)
        configs = []
        factory = StrategyFactory()
        default_llm_instruction = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content" # Shortened for brevity

        for name in strategy_names:
            try:
                strategy: Optional[ExtractionStrategy] = None
                if name == 'llm':
                    # Raises ConfigurationError if API key missing
                    strategy = factory.create_llm_strategy(
                        instruction=self.llm_instruction or default_llm_instruction,
                        api_key=self.llm_api_key,
                        model=self.llm_model
                    )
                elif name == 'css':
                    strategy = factory.create_css_strategy()
                elif name == 'xpath':
                    strategy = factory.create_xpath_strategy()
                elif name == 'no_extraction':
                    strategy = factory.create_no_extraction_strategy()
                elif name == 'cosine':
                    strategy = factory.create_cosine_strategy(semantic_filter=self.user_query, debug=self.debug)
                else:
                    logger.warning(f"Unknown extraction strategy name: '{name}'. Skipping.")
                    continue

                if strategy:
                    configs.append(ExtractionConfig(name=name, strategy=strategy))
            except ConfigurationError as e:
                 # Propagate config errors immediately during init
                 logger.error(f"Configuration error creating strategy '{name}': {e}. Failing scraper initialization.")
                 raise
            except Exception as e:
                # Treat unexpected strategy creation errors as configuration errors
                logger.error(f"Unexpected error creating strategy '{name}': {e}. Failing scraper initialization.", exc_info=True)
                raise ConfigurationError(f"Unexpected error creating strategy '{name}': {e}") from e

        if not configs:
             raise ConfigurationError("No valid extraction strategies could be created from the provided names.")

        return configs

    def _create_crawler_config(self) -> CrawlerRunConfig:
        """Creates default crawler configuration"""
        # Initialize without parameters, set attributes afterward
        config = CrawlerRunConfig()
        
        # Set attributes individually after initialization
        config.cache_mode = CacheMode.BYPASS
        
        # --- FIX: DO NOT set extraction_strategy here --- 
        # Let the markdown_generation_strategy handle populating markdown_content.
        # If specific extraction is needed later, it should be a separate step.
        logger.debug("CrawlerRunConfig will use basic config, not a specific extraction_strategy or markdown generator initially.")
        # config.extraction_strategy = [cfg.strategy for cfg in self.extraction_configs] # This was causing the error
        
        # Create and set markdown generator
        # --- FIX: DO NOT set markdown_generation_strategy here --- 
        # We will fetch HTML first, then decide how to process.
        # content_filter = PruningContentFilter()
        # config.markdown_generation_strategy = DefaultMarkdownGenerator(
        #     content_filter=content_filter
        # )
        
        return config

    async def scrape(self, url: str) -> Dict[str, ExtractionResult]:
        """
        Scrapes a single URL using the configured strategies asynchronously.

        Args:
            url: The URL to scrape.

        Returns:
            A dictionary where keys are strategy names and values are ExtractionResult objects
            containing the successfully extracted content.

        Raises:
            ValueError: If the URL is invalid.
            ScrapingError: If the crawling or extraction process fails critically.
            ConfigurationError: If underlying components (e.g., FastText, PDF parser) fail to load or configure.
        """
        if not url or not urlparse(url).scheme in ['http', 'https']:
            raise ValueError(f"Invalid URL provided for scraping: {url}")

        # --- Handle Special Cases --- #
        parsed_url = urlparse(url)
        if "wikipedia.org" in parsed_url.netloc:
            logger.info(f"Detected Wikipedia URL, using specialized extractor: {url}")
            try:
                content = await get_wikipedia_content(url, lang='en')
                return {
                    'wikipedia': ExtractionResult( # Use specific key
                        name='wikipedia',
                        content=content,
                        raw_markdown_length=len(content)
                    )
                }
            except (ValueError, ScrapingError, Exception) as e:
                # Catch broad Exception here just in case get_wiki raises something unexpected
                logger.error(f"Failed specialized Wikipedia extraction for {url}: {e}", exc_info=False)
                raise ScrapingError(f"Failed specialized Wikipedia extraction for {url}: {e}") from e

        if url.lower().endswith('.pdf'):
            logger.info(f"Detected PDF URL, attempting PDF extraction: {url}")
            try:
                content = await extract_from_pdf_url(url, debug=self.debug)
                return {
                    'pdf': ExtractionResult( # Use specific key
                        name='pdf',
                        content=content,
                        raw_markdown_length=len(content)
                    )
                }
            except (NotImplementedError, ConfigurationError, ScrapingError, Exception) as e:
                 logger.error(f"Failed PDF extraction for {url}: {e}", exc_info=False)
                 raise ScrapingError(f"Failed PDF extraction for {url}: {e}") from e

        # --- General Web Scraping using direct HTML fetching like oldscraper.py --- #
        try:
            logger.info(f"Fetching raw HTML for URL: {url}")
            
            # IMPORTANT: Use oldscraper's approach to fetch raw HTML directly
            raw_html = await self._fetch_raw_html(url)
            if not raw_html:
                raise ScrapingError(f"Failed to fetch raw HTML for URL: {url}")
            
            # Only log the length of HTML, not content
            html_length = len(raw_html)
            logger.info(f"Successfully fetched raw HTML for {url}, length: {html_length} chars")
            
            # Clean the HTML using the utility function
            cleaned_html = clean_html(raw_html, clean_svg=False, clean_base64=False)
            logger.debug(f"Cleaned HTML content length for {url}: {len(cleaned_html)} chars")
            
            final_content = cleaned_html
            
            if not final_content:
                # If content is empty after cleaning, treat as scraping error
                logger.warning(f"Cleaning resulted in empty content for URL: {url}")
                raise ScrapingError(f"HTML cleaning yielded empty content for URL: {url}")

            # Process results - use the name of the first configured strategy
            strategy_name_for_result = self.extraction_configs[0].name if self.extraction_configs else 'processed_content'

            results = {
                strategy_name_for_result: ExtractionResult(
                    name=strategy_name_for_result,
                    content=final_content,
                    raw_markdown_length=len(raw_html)
                )
            }
            
            logger.info(f"Successfully scraped and processed URL: {url}")
            return results

        except Exception as e:
            # Catch any errors during the scraping process
            # Avoid logging the full exception which might contain HTML
            error_msg = str(e)
            if len(error_msg) > 500:  # Truncate long error messages
                error_msg = error_msg[:500] + "... [truncated]"
            
            logger.error(f"Error scraping URL '{url}': {error_msg}", exc_info=False)
            if isinstance(e, ScrapingError):
                raise  # Re-raise ScrapingError
            raise ScrapingError(f"Failed to scrape URL '{url}': {error_msg}") from e

    async def _fetch_raw_html(self, url: str) -> Optional[str]:
        """Fetches raw HTML content using AsyncWebCrawler with minimal configuration."""
        try:
            # Use a simple config just to fetch HTML - EXACTLY like oldscraper.py
            config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=config)
                
            if hasattr(result, 'success') and result.success and hasattr(result, 'html'):
                # Don't log HTML content - can be too verbose and fill console
                html_length = len(result.html) if result.html else 0
                logger.debug(f"Fetched HTML for {url}: {html_length} bytes")
                return result.html
            else:
                error_msg = getattr(result, 'error', 'Unknown error')
                logger.error(f"Failed to fetch raw HTML for {url}. Error: {error_msg}")
                return None
        except Exception as e:
            logger.error(f"Exception during raw HTML fetch for {url}: {e}", exc_info=True)
            return None

    async def scrape_many(self, urls: List[str], sequential: bool = False) -> Dict[str, Dict[str, ExtractionResult]]:
        """Scrapes multiple URLs, optionally in sequence to avoid rate limits.
        
        Args:
            urls: List of URLs to scrape
            sequential: If True, scrape URLs sequentially to avoid rate limits
            
        Returns:
            Dictionary mapping URLs to their extraction results
        """
        if not urls:
            return {}
            
        results: Dict[str, Dict[str, ExtractionResult]] = {}
        exceptions_count = 0

        # Process sequentially if requested (to avoid rate limits)
        if sequential:
            logger.info(f"Processing {len(urls)} URLs sequentially to avoid rate limits")
            for url in urls:
                try:
                    result = await self.scrape(url)
                    results[url] = result
                except Exception as e:
                    exceptions_count += 1
                    log_level = logging.WARNING if isinstance(e, ScrapingError) else logging.ERROR
                    logger.log(log_level, f"Scraping failed for URL '{url}': {str(e)}", exc_info=False)
                    # Optional: Add delay between requests even on failure
                    await asyncio.sleep(1)  # 1 second delay between requests
        else:
            # Concurrent processing (original implementation)
            tasks = {url: asyncio.create_task(self.scrape(url)) for url in urls}
            
            # Wait for all tasks using asyncio.as_completed for better progress indication if needed
            for url, task in tasks.items():
                try:
                    result = await task
                    results[url] = result
                except Exception as e:
                    exceptions_count += 1
                    log_level = logging.WARNING if isinstance(e, ScrapingError) else logging.ERROR
                    logger.log(log_level, f"Scraping failed for URL '{url}' in batch: {type(e).__name__}: {e}", exc_info=False)

        if exceptions_count > 0:
            logger.warning(f"Completed scraping batch for {len(urls)} URLs with {exceptions_count} errors.")

        return results  # Return dict containing only successfully scraped URLs

    # extract method - Simplified, assuming called *after* successful crawl4ai run
    # This method seems less critical if crawl4ai handles the full pipeline
    async def extract(self, extraction_config: ExtractionConfig, url: str, pre_processed_content: str) -> ExtractionResult:
        """Extracts content using a specific strategy AFTER initial scrape. (Needs review based on usage)"""
        if not pre_processed_content:
            raise ValueError(f"Cannot run extraction strategy '{extraction_config.name}' on empty content for URL: {url}")

        logger.debug(f"Running post-crawl extraction strategy '{extraction_config.name}' for URL: {url}")
        try:
            # Determine if strategy execution is async or sync
            strategy = extraction_config.strategy
            execute_method = None
            if hasattr(strategy, 'execute_async'):
                execute_method = strategy.execute_async
            elif hasattr(strategy, 'execute'):
                 # Wrap sync method
                 loop = asyncio.get_running_loop()
                 execute_method = lambda content: loop.run_in_executor(None, strategy.execute, content)
            else:
                 raise NotImplementedError(f"Strategy '{extraction_config.name}' does not have a recognized execute method.")

            extracted_data = await execute_method(pre_processed_content)

            content_str = json.dumps(extracted_data) if isinstance(extracted_data, (dict, list)) else str(extracted_data)

            return ExtractionResult(
                name=extraction_config.name,
                content=content_str,
                # Lengths might be less relevant here
            )
        except LLMCommunicationError as e:
             logger.error(f"LLM error during '{extraction_config.name}' post-crawl strategy for {url}: {e}")
             raise ScrapingError(f"LLM failure in strategy '{extraction_config.name}' for {url}: {e}") from e
        except Exception as e:
            logger.error(f"Error executing post-crawl extraction strategy '{extraction_config.name}' for {url}: {e}", exc_info=True)
            raise ScrapingError(f"Failed post-crawl strategy '{extraction_config.name}' for {url}: {e}") from e

# Example usage (Optional, can be removed or placed under if __name__ == "__main__")
# async def main():
#     scraper = WebScraper(strategies=['no_extraction'], debug=True)
#     url = "https://example.com"
#     results = await scraper.scrape(url)
#     for name, result in results.items():
#         if result.success:
#             print(f"--- {name} Success ---")
#             print(result.content[:200] + "...")
#         else:
#             print(f"--- {name} Failed: {result.error} ---")
#
# if __name__ == "__main__":
#     asyncio.run(main()) 