from __future__ import annotations
import asyncio
import logging
from . import agents, schemas, helpers
from app.core.schemas import RunUsage

logger = logging.getLogger(__name__)

async def run_financial_analysis_orchestration(
    request: schemas.FinancialAnalysisRequest,
    agents_collection: agents.AgencyAgents,
    services: dict, # A dictionary of instantiated services
    update_callback: "FinancialsWebSocketUpdateHandler" 
) -> schemas.FinancialResearchResponse:
    """Orchestrates the entire financial analysis process."""
    
    usage_tracker = RunUsage()
    ticker = request.stock_symbol
    
    # 1. Parallel Data Fetching
    logger.info(f"Fetching all data for {ticker}...")
    if update_callback: await update_callback.data_fetching_start()
    
    fetched_data = await helpers.fetch_all_data_sources(
        symbol=ticker,
        config=agents_collection.config, # Assuming config is attached to agents_collection
        **services
    )
    
    # Destructure fetched data
    financials = fetched_data.get("financials")
    social_media_posts = fetched_data.get("social_media_posts")
    news_articles = fetched_data.get("news_articles")
    earnings_call_transcript = fetched_data.get("earnings_call_transcript")
    
    # 2. Parallel Analysis of Fetched Data
    logger.info(f"Data fetched. Starting parallel analysis for {ticker}...")
    if update_callback: await update_callback.analysis_start()
    
    analysis_tasks = []
    
    # Create tasks only if data is available
    if financials:
        # First, programmatically calculate key metrics
        key_metrics = helpers.calculate_key_metrics(financials)

        analysis_tasks.append(agents_collection.financials_agent.run(
            agents.FINANCIALS_AGENT_USER_TEMPLATE.format(
                ticker=ticker,
                key_metrics=agents.format_data_for_prompt(key_metrics),
                income_statement=agents.format_data_for_prompt(financials.income_statement),
                balance_sheet=agents.format_data_for_prompt(financials.balance_sheet),
                cash_flow=agents.format_data_for_prompt(financials.cash_flow)
            )
        ))
    if social_media_posts:
        analysis_tasks.append(agents_collection.sentiment_agent.run(
            agents.SENTIMENT_AGENT_USER_TEMPLATE.format(
                ticker=ticker,
                posts=agents.format_data_for_prompt(social_media_posts)
            )
        ))
    if news_articles:
        analysis_tasks.append(agents_collection.news_agent.run(
            agents.NEWS_AGENT_USER_TEMPLATE.format(
                ticker=ticker,
                articles=agents.format_data_for_prompt(news_articles)
            )
        ))
    if earnings_call_transcript:
        analysis_tasks.append(agents_collection.earnings_agent.run(
            agents.EARNINGS_CALL_AGENT_USER_TEMPLATE.format(
                ticker=ticker,
                transcript=agents.format_data_for_prompt(earnings_call_transcript)
            )
        ))

    # Run analysis agents concurrently
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    
    # Process results and update dashboard
    fundamental_analysis, sentiment_analysis, news_analysis, earnings_analysis = (None, None, None, None)
    
    # This is a bit brittle and depends on the order. A more robust solution might map task types to results.
    result_map = {
        schemas.FundamentalAnalysis: None,
        schemas.SentimentAnalysis: None,
        schemas.NewsAnalysis: None,
        schemas.EarningsCallAnalysis: None,
    }
    
    for result in analysis_results:
        if isinstance(result, Exception):
            logger.error(f"An analysis agent failed: {result}")
            continue
        
        # Update usage tracker
        usage_tracker.update_agent_usage(result.data.__class__.__name__, result.usage())
        
        for key_type in result_map.keys():
            if isinstance(result.data, key_type):
                result_map[key_type] = result.data
                if update_callback: await update_callback.send_analysis_update(result.data)
                break
                
    fundamental_analysis = result_map[schemas.FundamentalAnalysis]
    sentiment_analysis = result_map[schemas.SentimentAnalysis]
    news_analysis = result_map[schemas.NewsAnalysis]
    earnings_analysis = result_map[schemas.EarningsCallAnalysis]

    # 3. Final Report Generation
    logger.info(f"All analyses complete. Generating final report for {ticker}...")
    if update_callback: await update_callback.report_generation_start()

    report_generator_prompt = agents.REPORT_GENERATOR_USER_TEMPLATE.format(
        ticker=ticker,
        report_flavor=request.report_flavor,
        fundamental_analysis=agents.format_data_for_prompt(fundamental_analysis or "Not available."),
        sentiment_analysis=agents.format_data_for_prompt(sentiment_analysis or "Not available."),
        news_analysis=agents.format_data_for_prompt(news_analysis or "Not available."),
        earnings_call_analysis=agents.format_data_for_prompt(earnings_analysis or "Not available.")
    )

    final_report_result = await agents_collection.report_generator_agent.run(report_generator_prompt)
    usage_tracker.update_agent_usage("ReportGenerator", final_report_result.usage())
    
    final_report: schemas.FinancialReport = final_report_result.data
    
    logger.info(f"Financial analysis orchestration for {ticker} complete.")
    
    # 4. Final Response
    return schemas.FinancialResearchResponse(
        final_report=final_report,
        usage_statistics=usage_tracker.get_statistics()
    ) 