import logging
from pydantic import HttpUrl, ValidationError
from typing import List

from .config import CPEConfig
from .schemas import CPERequest, CPEResponse, ExtractedCompanyData, CompanyProfile
from .helpers import group_by_domain, aggregate_html
from .agents import get_cpe_agents
from .prompts import EXTRACTOR_USER_MESSAGE_TEMPLATE
from app.agencies.services.scraper_utils.emailfinder import find_emails_deep

logger = logging.getLogger(__name__)

async def run_cpe(request: CPERequest, config: CPEConfig) -> CPEResponse:
    """
    Executes the Company Profile Extractor workflow for given start URLs.
    """
    agents = get_cpe_agents(config)
    profiles: List[CompanyProfile] = []

    for url in request.start_urls:
        logger.info(f"Processing start URL: {url}")
        pages, _ = await find_emails_deep(
            str(url),
            max_depth=config.max_crawl_depth,
            max_pages=config.max_crawl_pages
        )
        domain_groups = group_by_domain(pages)
        logger.info(f"Found {len(domain_groups)} domains after crawling {url}")

        for domain_str, pages_list in domain_groups.items():
            logger.info(f"Processing domain: {domain_str}")
            html_blob = aggregate_html(pages_list, max_bytes=200000)
            
            if not html_blob.strip():
                logger.warning(f"Skipping domain {domain_str} due to empty aggregated HTML.")
                continue

            user_prompt = EXTRACTOR_USER_MESSAGE_TEMPLATE.format(
                html_blob=html_blob
            )
            
            try:
                logger.debug(f"Calling extractor agent for domain: {domain_str}")
                result = await agents.extractor.run(user_prompt)
                extracted_data: ExtractedCompanyData = result.data
                logger.debug(f"Extractor agent returned data for {domain_str}")

                try:
                    domain_url = HttpUrl(f"https://{domain_str}")
                except ValidationError as url_val_error:
                    logger.error(f"Failed to construct valid HttpUrl for domain '{domain_str}': {url_val_error}")
                    continue

                profile = CompanyProfile(
                    domain=domain_url,
                    **extracted_data.model_dump()
                )
                profiles.append(profile)
                logger.info(f"Successfully created CompanyProfile for domain: {domain_str}")

            except Exception as agent_error:
                logger.error(f"Extractor agent failed for domain {domain_str}: {agent_error}", exc_info=True)
                continue

    usage_stats = {"profiles_extracted": len(profiles)}
    logger.info(f"CPE run complete. Extracted {len(profiles)} profiles.")
    return CPEResponse(profiles=profiles, usage_statistics=usage_stats) 