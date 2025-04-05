import asyncio
import argparse
import logging
import json
from dataclasses import asdict
from typing import Optional

# Adjust the import path based on running from the workspace root
from app.services.scraping import WebScraper, ExtractionResult

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress noisy logs from dependencies if needed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)

async def main(urls_to_scrape: list[str], debug_mode: bool, download_pdfs: bool):
    """Runs the scraper for the given URLs and prints results."""
    logger = logging.getLogger("test_scrape")
    logger.info(f"Starting scrape test for {len(urls_to_scrape)} URLs.")
    if debug_mode:
        logger.parent.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    # Initialize the scraper
    scraper = WebScraper(debug=debug_mode)

    try:
        # Run scrape_many (concurrently by default)
        results = await scraper.scrape_many(
            urls=urls_to_scrape,
            download_pdfs=download_pdfs,
            pdf_save_dir="downloaded_pdfs_test" # Use a test-specific dir
        )

        print("\n--- Scraping Results ---")
        for url, result_obj in results.items():
            print(f"\nURL: {url}")
            if isinstance(result_obj, ExtractionResult):
                result_dict = asdict(result_obj)
                content_value = result_dict.get('content')

                # Ensure content_value is a string before processing
                content_str: Optional[str] = None
                if isinstance(content_value, str):
                    content_str = content_value
                elif content_value is not None:
                    # If it's not a string, try converting it using str()
                    # This should handle MarkdownGenerationResult via its __str__ method
                    logger.debug(f"Content for {url} is type {type(content_value)}, converting to string.")
                    try:
                         content_str = str(content_value)
                    except Exception as e:
                        logger.error(f"Failed to convert content to string for {url}: {e}")
                        content_str = "[Error converting content to string]"
                else:
                    # Handle case where content is None
                    content_str = None

                # Update the dictionary with the guaranteed string (or None)
                result_dict['content'] = content_str

                # Now process the dictionary, checking if content_str is not None
                if result_dict.get('name') == 'pdf':
                    print(f"  Source: pdf")
                    print(f"  Content: [PDF Content Extracted - Suppressed in Output]")
                    print(f"  Raw Markdown Length: {result_dict.get('raw_markdown_length', 'N/A')}")
                else:
                    # Truncate long content *if* content_str is not None
                    if content_str and len(content_str) > 500:
                        result_dict['content'] = content_str[:500] + "... [truncated]"
                    # Print the potentially modified dictionary
                    print(json.dumps(result_dict, indent=2))
            else:
                # Should not happen with current scrape_many logic, but good practice
                print(f"  Unexpected result type: {type(result_obj)}")
                print(f"  Result: {result_obj}")

        success_count = len(results)
        failure_count = len(urls_to_scrape) - success_count
        logger.info(f"Scraping finished. Success: {success_count}, Failures (logged): {failure_count}")

    except Exception as e:
        logger.critical(f"An error occurred during the scrape_many execution: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the WebScraper service.")
    parser.add_argument(
        "--urls",
        nargs='+',
        required=True,
        help="List of URLs to scrape."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for scraper and test script."
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Save downloaded PDFs to 'downloaded_pdfs_test' directory."
    )

    args = parser.parse_args()

    asyncio.run(main(args.urls, args.debug, args.download_pdfs)) 