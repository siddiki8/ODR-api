"""
Service for scraping web content using Crawl4AI and various strategies.
Consolidates functionality previously spread across multiple files.
"""

import asyncio
import os
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

# External Dependencies (Ensure these are in requirements.txt)
import fasttext
import wikipediaapi
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
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

# --- Dependencies moved from extraction_result.py ---
@dataclass
class ExtractionResult:
    """Holds the results of an extraction operation"""
    name: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
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
    ) -> LLMExtractionStrategy:
        # TODO: Consider moving LLM provider/key details to config
        load_dotenv() # Ensure env vars are loaded for API key
        return LLMExtractionStrategy(
            input_format=input_format,
            provider="openrouter/google/gemini-flash-1.5",  # Updated model, make configurable?
            api_token=os.getenv("OPENROUTER_API_KEY"),
            instruction=instruction
        )

    @staticmethod
    def create_css_strategy() -> JsonCssExtractionStrategy:
        # Default schema, might need to be configurable
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
         # Default schema, might need to be configurable
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

def _load_fasttext_model():
    """Loads the FastText model if not already loaded."""
    global _fasttext_model
    if _fasttext_model is None:
        print("Loading FastText quality model...") # Add log for visibility
        try:
            model_path = hf_hub_download(
                "kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2",
                "model.bin"
            )
            _fasttext_model = fasttext.load_model(model_path)
            print("FastText quality model loaded.")
        except Exception as e:
            print(f"Error loading FastText model: {e}. Quality filtering may not work.")
            # Handle error appropriately - maybe raise, or set model to a dummy?
            # For now, it will remain None, and predict will handle it.

def replace_newlines(text: str) -> str:
    """Replace multiple newlines with a single space."""
    return re.sub("\\n+", " ", text)

score_dict = {
    '__label__': 0,
    '__label__Low': 0,
    '__label__Mid': 1,
    '__label__High': 2
}

def predict_educational_value(text_list: List[str]) -> List[float]:
    """
    Predict educational value scores for a list of texts.
    Returns a list of scores between 0 and 2.
    Loads the model lazily on first call.
    """
    # Ensure the model is loaded
    _load_fasttext_model()

    # Check if model loading failed
    if _fasttext_model is None:
        print("Warning: FastText model not loaded. Returning default scores (0.0).")
        return [0.0] * len(text_list)

    text_list = [replace_newlines(text) for text in text_list]
    pred = _fasttext_model.predict(text_list, k=-1)
    score_list = []
    for labels, scores in zip(*pred):
        score = 0
        for label, score_val in zip(labels, scores):
            # Check if label exists in dict, default to 0 if not
            score += score_dict.get(label, 0) * score_val
        score_list.append(float(score))
    return score_list

def clean_markdown_links(text: str, min_quality_score: float = 0.2) -> Tuple[str, float]:
    """
    Clean markdown links and filter low-quality content.
    Returns tuple of (cleaned_text, quality_score)
    """
    # Split by double newlines to preserve paragraph structure
    paragraphs = text.split('\\n\\n')

    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Preserve code blocks by checking if paragraph contains ``` tags
        if '```' in paragraph:
            cleaned_paragraphs.append(paragraph)
            continue

        lines = paragraph.split('\\n')
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
            cleaned_paragraphs.append('\\n'.join(filtered_lines))

    # Rejoin with double newlines
    cleaned_text = '\\n\\n'.join(cleaned_paragraphs)

    # Get quality score
    quality_score = predict_educational_value([cleaned_text])[0]

    return cleaned_text, quality_score

def filter_quality_content(text: str, min_quality_score: float = 0.2) -> str:
    """
    Filter content based on quality and returns concatenated quality content
    """
    # Split text into paragraphs
    paragraphs = text.split('\\n\\n')

    # Process each paragraph
    quality_content = []
    for paragraph in paragraphs:
        if not paragraph.strip():  # Skip empty paragraphs
            continue

        cleaned_text, quality_score = clean_markdown_links(paragraph, min_quality_score)
        if cleaned_text and quality_score >= min_quality_score:
            quality_content.append((cleaned_text, quality_score))

    # Debug print
    # print(f"Found {len(quality_content)} quality paragraphs out of {len(paragraphs)} total") # Optional debug

    if quality_content:
        return "\\n\\n".join(text for text, _ in quality_content)
    return text  # Return original text if no quality content found

def get_wikipedia_content(url: str) -> str | None:
    """
    Extract content from a Wikipedia URL.

    Args:
        url: Wikipedia URL to scrape

    Returns:
        str: Page content if found, None otherwise
    """
    wiki = wikipediaapi.Wikipedia(user_agent="deep_research_api", language='en') # Changed user agent

    # Extract the page title from URL (everything after /wiki/)
    try:
        title = url.split('/wiki/')[-1]
        page = wiki.page(title)
        if page.exists():
            return page.text
        return None
    except Exception:
        return None

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
    html = re.sub(r'\\n\\s*\\n', '\\n', html)
    html = re.sub(r'\\s+', ' ', html)

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

# PDF extraction for academic papers
async def extract_from_pdf_url(url: str, debug: bool = False) -> Optional[str]:
    """
    Attempts to extract text from a PDF URL using an external service.
    Falls back to simpler methods if needed.
    """
    # First try direct download and processing if PDF
    if url.lower().endswith('.pdf'):
        try:
            if debug:
                print(f"Debug: Attempting PDF extraction for {url}")
            
            # Here you would typically use a PDF extraction library
            # For now, we'll use a simplified approach with requests
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Basic text extraction - in production you'd use a proper PDF parser
                # like PyPDF2, pdfminer.six, or an external API
                text = response.text[:10000]  # Just grab some content for demonstration
                return f"[PDF Content from {url}]\n\n{text}"
        except Exception as e:
            if debug:
                print(f"Debug: PDF extraction failed for {url}: {e}")
    
    # For arXiv papers, try to get the abstract
    if 'arxiv.org' in url:
        arxiv_id = None
        # Extract arXiv ID from URL
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url)
        if match:
            arxiv_id = match.group(1)
            
        if arxiv_id:
            try:
                arxiv_api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
                response = requests.get(arxiv_api_url, timeout=15)
                if response.status_code == 200:
                    # Parse the XML response
                    soup = BeautifulSoup(response.text, 'xml')
                    title = soup.find('title').text
                    abstract = soup.find('summary').text
                    authors = [author.find('name').text for author in soup.find_all('author')]
                    author_text = ", ".join(authors)
                    
                    return f"Title: {title}\nAuthors: {author_text}\nAbstract: {abstract}"
            except Exception as e:
                if debug:
                    print(f"Debug: arXiv API extraction failed for {url}: {e}")
    
    return None

# --- Original WebScraper class (from crawl4ai_scraper.py) with updated imports ---
class WebScraper:
    """Unified scraper that encapsulates all extraction strategies and configuration"""
    def __init__(
        self,
        browser_config: Optional[BrowserConfig] = None,
        strategies: List[str] = ['no_extraction'],
        llm_instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
        user_query: Optional[str] = None, # Note: user_query is not used in this version
        debug: bool = False,
        filter_content: bool = False # Note: filter_content not fully implemented in extract method yet
    ):
        self.browser_config = browser_config or BrowserConfig(headless=True, verbose=debug)
        self.debug = debug
        self.factory = StrategyFactory() # Use the factory defined above
        self.strategies = strategies or ['no_extraction'] # Simplified default
        self.llm_instruction = llm_instruction
        # self.user_query = user_query # Currently unused
        self.filter_content = filter_content # Currently unused in extract

        # Validate strategies
        valid_strategies = {'markdown_llm', 'html_llm', 'fit_markdown_llm', 'css', 'xpath', 'no_extraction', 'cosine'}
        invalid_strategies = set(self.strategies) - valid_strategies
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")

        # Initialize strategy map using the factory methods defined above
        self.strategy_map = {
            'markdown_llm': lambda: self.factory.create_llm_strategy('markdown', self.llm_instruction),
            'html_llm': lambda: self.factory.create_llm_strategy('html', self.llm_instruction),
            'fit_markdown_llm': lambda: self.factory.create_llm_strategy('fit_markdown', self.llm_instruction),
            'css': self.factory.create_css_strategy,
            'xpath': self.factory.create_xpath_strategy,
            'no_extraction': self.factory.create_no_extraction_strategy,
            'cosine': lambda: self.factory.create_cosine_strategy(debug=self.debug)
        }

    def _create_crawler_config(self) -> CrawlerRunConfig:
        """Creates default crawler configuration"""
        content_filter = PruningContentFilter()
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=content_filter
            )
        )

    async def scrape(self, url: str) -> Dict[str, ExtractionResult]:
        """
        Scrape URL using configured strategies

        Args:
            url: Target URL to scrape
        """
        results = {}

        # Handle academic paper URLs with specialized extraction
        if is_academic_url(url):
            if self.debug:
                print(f"Debug: Detected academic URL: {url}")
            
            # Try specialized academic extraction first
            academic_content = await extract_from_pdf_url(url, self.debug)
            
            if academic_content:
                # If we got content from specialized extraction, use it for all strategies
                if self.debug:
                    print(f"Debug: Successfully extracted academic content using specialized method")
                return {
                    strategy_name: ExtractionResult(
                        name=strategy_name, success=True, content=academic_content
                    ) for strategy_name in self.strategies
                }

        # Handle Wikipedia URLs separately using the utility function
        if 'wikipedia.org/wiki/' in url:
            try:
                content = get_wikipedia_content(url) # Use utility function
                # Apply quality filter if enabled
                # Note: filter_content currently only affects non-Wikipedia flow
                # if self.filter_content and content: ...
                if not content:
                    return {
                        strategy_name: ExtractionResult(
                            name=strategy_name, success=False, error="Wikipedia content empty or filtered."
                        ) for strategy_name in self.strategies
                    }
                # Return success for all strategies for Wikipedia content
                return {
                    strategy_name: ExtractionResult(
                        name=strategy_name, success=True, content=content
                    ) for strategy_name in self.strategies
                }
            except Exception as e:
                if self.debug:
                    print(f"Debug: Wikipedia extraction attempt failed: {str(e)}, falling back to standard scrape.")
                # Fall through to normal scraping

        # Normal scraping for non-Wikipedia URLs
        raw_html = await self._fetch_raw_html(url)
        if raw_html is None:
             return {
                strategy_name: ExtractionResult(
                    name=strategy_name, success=False, error="Failed to fetch raw HTML content."
                ) for strategy_name in self.strategies
            }

        # Clean the HTML *before* passing to strategies
        # TODO: Revisit if CSS/XPath need raw HTML (they likely do)
        cleaned_html = clean_html(raw_html) # Use utility function

        # TODO: Implement filter_content logic here if needed on cleaned_html
        base_content_for_strategies = cleaned_html
        # if self.filter_content: ...

        # Run extraction strategies
        for strategy_name in self.strategies:
            # Check if strategy exists in map
            strategy_func = self.strategy_map.get(strategy_name)
            if not strategy_func:
                 results[strategy_name] = ExtractionResult(name=strategy_name, success=False, error=f"Strategy '{strategy_name}' not found in map.")
                 continue

            config = ExtractionConfig( # Use config class defined above
                name=strategy_name,
                strategy=strategy_func()
            )
            # Pass the prepared base content to extract method
            # NOTE: The `extract` method needs review for compatibility with this flow
            result = await self.extract(config, url, base_content_for_strategies)
            results[strategy_name] = result

        return results

    async def scrape_many(self, urls: List[str]) -> Dict[str, Dict[str, ExtractionResult]]:
        """
        Scrape multiple URLs using configured strategies in parallel

        Args:
            urls: List of target URLs to scrape

        Returns:
            Dictionary mapping URLs to their extraction results
        """
        tasks = [self.scrape(url) for url in urls]
        results_list = await asyncio.gather(*tasks)
        return {url: result for url, result in zip(urls, results_list)}

    async def _fetch_raw_html(self, url: str) -> Optional[str]:
        """Fetches raw HTML content using AsyncWebCrawler."""
        try:
            # Use a simple config just to fetch HTML
            config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=config)
            if result.success and hasattr(result, 'html'):
                return result.html
            else:
                if self.debug:
                    print(f"Debug: Failed to fetch raw HTML for {url}. Error: {getattr(result, 'error', 'Unknown')}")
                return None
        except Exception as e:
            if self.debug:
                print(f"Debug: Exception during raw HTML fetch for {url}: {e}")
            return None

    async def extract(self, extraction_config: ExtractionConfig, url: str, pre_processed_content: str) -> ExtractionResult:
        """
        Internal method to perform extraction using specified strategy on pre-processed content.
        NOTE: This method bypasses rerunning Crawl4AI and operates on the given string.
              This breaks strategies relying on DOM (CSS/XPath) or those needing Crawl4AI's
              internal handling (LLM strategies might need adaptation).
              Currently, only 'no_extraction' and 'cosine' reliably work with this flow.
        """
        strategy_name = extraction_config.name
        content = None
        success = False # Default to False unless explicitly set
        error = None
        raw_markdown_length = 0
        citations_markdown_length = 0

        try:
            if strategy_name in ['no_extraction', 'cosine']:
                # These strategies can potentially work with the pre-processed string
                content = pre_processed_content
                raw_markdown_length = len(pre_processed_content)
                success = True
                if self.debug:
                    print(f"Debug: Using pre-processed content for strategy: {strategy_name}")
            elif strategy_name in ['markdown_llm', 'html_llm', 'fit_markdown_llm']:
                # LLM strategies currently require Crawl4AI's execution context
                error = f"LLM strategy '{strategy_name}' direct invocation on pre-processed content not supported by this simplified extract method."
                if self.debug:
                    print(f"Debug: {error}")
            elif strategy_name in ['css', 'xpath']:
                # CSS/XPath need the DOM, not just a cleaned HTML string.
                error = f"Strategy '{strategy_name}' requires DOM access, incompatible with pre-processing flow in this simplified extract method."
                if self.debug:
                    print(f"Debug: {error}")
            else:
                error = f"Unknown or unsupported strategy '{strategy_name}' in simplified extract method."
                if self.debug:
                    print(f"Debug: {error}")

        except Exception as e:
            success = False
            error = f"Exception during simplified extract for strategy {strategy_name}: {str(e)}"
            if self.debug:
                print(f"Debug: {error}")


        return ExtractionResult( # Use result class defined above
            name=strategy_name,
            success=success,
            content=content,
            error=error,
            raw_markdown_length=raw_markdown_length,
            citations_markdown_length=citations_markdown_length # Typically 0 in this simplified flow
        )

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