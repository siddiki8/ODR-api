import logging
import wikipediaapi
from urllib.parse import urlparse

# Import ScrapingError relative to the app root
from app.core.exceptions import ScrapingError

logger = logging.getLogger(__name__)

# Initialize Wikipedia API
# Use a custom User-Agent as per Wikipedia API guidelines
# See: https://www.mediawiki.org/wiki/API:Etiquette#User-Agent_header
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='ODR_API_Scraper/1.0 (https://example.com/contact; email@example.com)', # Replace with actual contact info if possible
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

async def get_wikipedia_content(url: str, lang: str = 'en') -> str:
    """
    Fetches and extracts content from a Wikipedia page URL.

    Args:
        url: The full URL of the Wikipedia page.
        lang: The language code for Wikipedia (default: 'en').

    Returns:
        The main text content of the Wikipedia page.

    Raises:
        ScrapingError: If the URL is invalid, the page doesn't exist,
                       or any other error occurs during fetching/parsing.
    """
    logger.info(f"Attempting to fetch Wikipedia content for URL: {url} with lang: {lang}")
    try:
        parsed_url = urlparse(url)
        if not parsed_url.netloc.endswith("wikipedia.org"):
            raise ScrapingError(f"Invalid URL: Not a Wikipedia domain - {url}")
        page_title = parsed_url.path.split('/')[-1]
        if not page_title:
             raise ScrapingError(f"Could not extract page title from URL: {url}")

        if lang != wiki_wiki.language:
            wiki_wiki.language = lang
            logger.info(f"Switched Wikipedia API language to: {lang}")

        page = wiki_wiki.page(page_title)

        if not page.exists():
            logger.warning(f"Wikipedia page not found: {page_title} (URL: {url})")
            raise ScrapingError(f"Wikipedia page not found: {page_title}")

        # Handle disambiguation pages (check summary)
        page_summary = page.summary # Get summary once
        if "may refer to:" in page_summary.lower() or "can refer to:" in page_summary.lower():
             logger.warning(f"'{page_title}' is likely a disambiguation page (URL: {url}). Skipping detailed content extraction.")
             return page_summary # Return summary for disambiguation

        logger.info(f"Successfully fetched Wikipedia page: {page.title}")
        content = page.text
        if not content:
            logger.warning(f"Extracted empty content from Wikipedia page: {page.title} (URL: {url})")
            raise ScrapingError(f"Extracted empty content from Wikipedia page: {page.title}")

        return content

    # Simplified exception handling for wikipediaapi errors
    except Exception as e:
        # Check if it's likely a wikipediaapi specific error or a general one
        if "wikipediaapi" in str(type(e)).lower(): # Basic check
             logger.error(f"Wikipedia API error for {url}: {e}", exc_info=True)
             raise ScrapingError(f"Wikipedia API error for {url}: {e}") from e
        else:
            # Handle other unexpected errors
            logger.error(f"Unexpected error fetching Wikipedia content for {url}: {e}", exc_info=True)
            raise ScrapingError(f"Unexpected error fetching Wikipedia content for {url}: {e}") from e