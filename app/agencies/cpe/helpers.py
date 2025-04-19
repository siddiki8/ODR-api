from typing import List, Dict
from urllib.parse import urlparse
from app.agencies.services.scraper_utils.emailfinder import EmailPageResult


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