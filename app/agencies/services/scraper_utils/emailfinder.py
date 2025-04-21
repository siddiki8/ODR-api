"""
Utility to perform deep crawling on a website to find email addresses.
"""

import asyncio
import re
import logging
from typing import List, Dict, Optional, TypedDict, Tuple

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    CacheMode
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
# Use LXMLWebScrapingStrategy explicitly if default doesn't guarantee HTML in result.content or result.html
# from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define a structure for the results
class EmailPageResult(TypedDict):
    url: str
    html: str
    emails: List[str]

# Regex pattern to find email addresses (adjust as needed for robustness)
# This is a common pattern, but might miss some edge cases or have false positives.
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

async def find_emails_deep(
    start_url: str,
    max_depth: int = 1, # Default to crawl start_url + 1 level deep
    max_pages: Optional[int] = 50 # Limit total pages crawled
) -> Tuple[List[EmailPageResult], List[str]]:
    """
    Performs a deep crawl starting from `start_url` to find pages containing email addresses.

    Args:
        start_url: The URL to begin crawling from.
        max_depth: The maximum depth to crawl beyond the start URL. (0 means only the start URL)
        max_pages: The maximum total number of pages to crawl.

    Returns:
        A tuple containing:
        - A list of dictionaries (EmailPageResult) for pages where emails were detected.
        - A list of all URLs crawled during the process.
    """
    results: List[EmailPageResult] = []
    all_crawled_urls: List[str] = []
    email_pattern = re.compile(EMAIL_REGEX)
    crawled_count = 0
    # --- Add set to track seen emails across this crawl ---
    seen_emails_overall: set[str] = set()

    logger.info(f"Starting deep email search on {start_url} (max_depth={max_depth}, max_pages={max_pages})")

    try:
        # Configure the deep crawl strategy
        # BFS explores level by level
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False, # Stay within the same domain
            max_pages=max_pages
        )

        # Configure the run settings
        # Stream=True processes results as they arrive
        # We need HTML, so ensure the scraping strategy provides it.
        # Default might be sufficient, but explicitly setting LXML ensures access to result.html
        config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl_strategy,
            # scraping_strategy=LXMLWebScrapingStrategy(), # Uncomment if default doesn't work
            cache_mode=CacheMode.BYPASS, # Bypass cache entirely
            stream=True,
            verbose=True # Set to True for detailed Crawl4AI logs
        )

        # Initialize and run the crawler
        async with AsyncWebCrawler() as crawler:
            # arun returns an async iterator when stream=True
            async for result in await crawler.arun(start_url, config=config):
                crawled_count += 1
                all_crawled_urls.append(result.url) # Store every crawled URL

                # --- Debug Logging for Specific Page ---
                if "/our-team" in result.url:
                    logger.info(f"DEBUG: HTML content received for {result.url} starts with:\n{result.html[:500] if result.html else '[No HTML content]'}\n---")
                # --- End Debug Logging ---

                if result.success and result.html:
                    # Search for emails in the raw HTML content
                    found_emails_on_page = set(email_pattern.findall(result.html))

                    if found_emails_on_page:
                        # --- Check for NEW emails not seen before in this crawl ---
                        newly_found_emails = found_emails_on_page - seen_emails_overall

                        if newly_found_emails:
                            logger.info(f"Found {len(newly_found_emails)} new emails on: {result.url}")
                            page_result: EmailPageResult = {
                                "url": result.url,
                                "html": result.html, # Include HTML because new emails were found
                                "emails": list(newly_found_emails) # Store only the newly found unique emails
                            }
                            results.append(page_result)
                            # Add the newly found emails to the overall set for this crawl
                            seen_emails_overall.update(newly_found_emails)
                        else:
                            # Emails found, but all were duplicates seen on previous pages in this crawl
                            logger.debug(f"Found only duplicate emails ({len(found_emails_on_page)}) on: {result.url}. Skipping HTML storage.")
                            # Optional: Could still append a result with html=None if needed elsewhere
                            # page_result: EmailPageResult = {"url": result.url, "html": None, "emails": list(found_emails_on_page)}
                            # results.append(page_result)
                    else:
                         logger.debug(f"No emails found on: {result.url}")

                elif not result.success:
                    logger.warning(f"Failed to crawl {result.url}: {result.error_message}")
                else:
                    logger.debug(f"Crawled {result.url} but no HTML content available.")

                # Optional: Early exit if max_pages is hit by the stream logic itself
                # (Though BFSDeepCrawlStrategy with max_pages should handle this)
                # if max_pages is not None and crawled_count >= max_pages:
                #     logger.info(f"Reached max_pages limit ({max_pages}). Stopping crawl.")
                #     break # Manually break needed if strategy doesn't stop stream exactly

    except Exception as e:
        logger.error(f"An error occurred during the deep crawl from {start_url}: {e}", exc_info=True)

    logger.info(f"Finished deep email search. Found new emails on {len(results)} out of {crawled_count} crawled pages.")
    return results, all_crawled_urls

# Example usage (for testing purposes)
async def main():
    # Replace with a URL you want to test (be mindful of website terms of service)
    # test_url = "https://example.com" # Use a site likely to have contact info / less complex structure
    test_url = "https://roundtableadvisory.com/" # User requested test URL
    # test_url = "https://www.google.com" # Less likely to have direct emails on crawlable pages

    # Example using a local file server (if you have one running)
    # Make sure the server allows directory listing or has links
    # test_url = "http://localhost:8000/"

    print(f"--- Running Email Finder Test on {test_url} ---")
    # Adjust call to unpack the tuple
    found_pages, all_urls = await find_emails_deep(test_url, max_depth=1, max_pages=20)

    if found_pages:
        print(f"\n--- Found Emails on {len(found_pages)} Pages ---")
        for page in found_pages:
            print(f"URL: {page['url']}")
            print(f"  Emails: {page['emails']}")
            # print(f"  HTML Snippet: {page['html'][:200]}...") # Uncomment to see HTML start
            print("-" * 20)
    else:
        print("\n--- No pages with emails found within the crawl limits. ---")

    # Print all crawled URLs
    print(f"\n--- Total URLs Crawled ({len(all_urls)}) ---")
    if all_urls:
        for url in all_urls:
            print(url)
    else:
        print("No URLs were crawled.")

if __name__ == "__main__":
    # Setup basic logging for the test run
    # logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
