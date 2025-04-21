import logging
from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urlparse
from pydantic import HttpUrl, ValidationError
from app.agencies.services.scraper_utils.emailfinder import find_emails_deep, EmailPageResult
from app.agencies.services.search import SearchResult
from .config import CPEConfig
from .schemas import ExtractedCompanyData, CompanyProfile

logger = logging.getLogger(__name__)


def group_by_domain(pages: List[EmailPageResult]) -> Dict[str, List[EmailPageResult]]:
    """
    Bucket pages by their base domain (netloc).
    """
    groups: Dict[str, List[EmailPageResult]] = {}
    for page in pages:
        parsed = urlparse(page['url'])
        domain = parsed.netloc
        groups.setdefault(domain, []).append(page)
    return groups


def aggregate_html(pages: List[EmailPageResult], max_bytes: int = 50000) -> str:
    """
    Concatenate HTML snippets from pages, truncating each to max_bytes to respect token limits.
    """
    snippets = []
    for page in pages:
        html = page.get('html', '') or ''
        # truncate each page's HTML to first max_bytes chars
        snippets.append(html[:max_bytes])
    return '\n\n'.join(snippets)


def filter_and_validate_search_results(
    search_results_map: Dict[str, List[SearchResult]]
) -> Tuple[List[SearchResult], Set[str]]:
    """
    Filters search results for unique, valid URLs and returns the results and seen links.
    """
    all_search_results: List[SearchResult] = []
    seen_links: Set[str] = set()
    for task_query, results in search_results_map.items():
        for result in results:
            if hasattr(result, 'link') and result.link and result.link not in seen_links:
                try:
                    validated_link = str(HttpUrl(result.link))
                    all_search_results.append(result)
                    seen_links.add(validated_link)
                except ValidationError as link_val_error:
                    logger.warning(f"Skipping invalid search result link '{result.link}': {link_val_error}")
            elif not hasattr(result, 'link') or not result.link:
                logger.debug(f"Skipping search result without a valid link: {result}")
    logger.info(f"Search tasks yielded {len(all_search_results)} unique valid results.")
    return all_search_results, seen_links


async def crawl_for_email_pages(
    start_url: str,
    config: CPEConfig
) -> Optional[List[EmailPageResult]]:
    """
    Crawls a starting URL to find pages with new emails using find_emails_deep.
    Returns the list of pages or None if crawling fails.
    """
    try:
        # find_emails_deep now only returns pages with new emails and their HTML
        pages_with_new_emails, _ = await find_emails_deep(
            start_url,
            max_depth=config.max_crawl_depth,
            max_pages=config.max_crawl_pages
        )
        return pages_with_new_emails
    except Exception as crawl_err:
        logger.error(f"Crawling/Email finding failed for start URL {start_url}: {crawl_err}", exc_info=True)
        return None


def make_company_profile(
    domain: str,
    extracted_data: ExtractedCompanyData
) -> Optional[CompanyProfile]:
    """
    Creates a CompanyProfile object from extracted data and domain.
    Returns the profile or None if domain URL validation fails.
    """
    try:
        # Use the derived domain for the profile, ensuring it's a valid URL
        domain_url_str = f"https://{domain}" # Assume HTTPS
        domain_url = HttpUrl(domain_url_str)
        
        profile = CompanyProfile(
            domain=domain_url,
            **extracted_data.model_dump()
        )
        logger.info(f"Successfully created CompanyProfile for domain: {domain}")
        return profile
    except ValidationError as url_val_error:
        logger.error(f"Failed to construct valid HttpUrl for domain '{domain}' (derived from {domain_url_str}): {url_val_error}")
        return None 