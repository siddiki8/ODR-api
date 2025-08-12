from __future__ import annotations
import asyncio
import logging
from . import agents, schemas, helpers
from app.core.schemas import RunUsage, UsageStatistics

logger = logging.getLogger(__name__)

async def run_financial_analysis_orchestration(
    request: schemas.FinancialAnalysisRequest,
    agents_collection: agents.AgencyAgents,
    config: schemas.FinancialResearchConfig, # Explicitly pass config
    services: dict, 
    update_callback: None, 
) -> schemas.FinancialResearchResponse:
    """Orchestrates the entire financial analysis process."""
    
    usage_tracker = RunUsage()
    ticker = request.stock_symbol
    
    # 1. Parallel Data Fetching
    logger.info(f"Fetching all data for {ticker}...")
    if update_callback: await update_callback.data_fetching_start()
    
    fetched_data = await helpers.fetch_all_data_sources(
        symbol=ticker,
        config=config, # Pass the explicit config
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
    
    # Define tasks with identifying names
    tasks = {}
    if financials:
        key_metrics = helpers.calculate_key_metrics(financials)
        tasks["financials"] = agents_collection.financials_agent.run(
            agents.FINANCIALS_AGENT_USER_TEMPLATE.format(
                ticker=ticker,
                key_metrics=agents.format_data_for_prompt(key_metrics),
                income_statement=agents.format_data_for_prompt(financials.income_statement),
                balance_sheet=agents.format_data_for_prompt(financials.balance_sheet),
                cash_flow=agents.format_data_for_prompt(financials.cash_flow)
            )
        )
    if social_media_posts:
        tasks["sentiment"] = agents_collection.sentiment_agent.run(
            agents.SENTIMENT_AGENT_USER_TEMPLATE.format(ticker=ticker, posts=agents.format_data_for_prompt(social_media_posts))
        )
    if news_articles:
        tasks["news"] = agents_collection.news_agent.run(
            agents.NEWS_AGENT_USER_TEMPLATE.format(ticker=ticker, articles=agents.format_data_for_prompt(news_articles))
        )
    if earnings_call_transcript:
        tasks["earnings"] = agents_collection.earnings_agent.run(
            agents.EARNINGS_CALL_AGENT_USER_TEMPLATE.format(ticker=ticker, transcript=agents.format_data_for_prompt(earnings_call_transcript))
        )

    # Run analysis agents concurrently using a dictionary
    task_results = {name: asyncio.create_task(coro) for name, coro in tasks.items()}
    await asyncio.gather(*task_results.values(), return_exceptions=True)

    # Process results more robustly
    analysis_outputs = {}
    for name, task in task_results.items():
        try:
            result = task.result()
            analysis_outputs[name] = result.data
            usage_tracker.update_agent_usage(f"{name}_agent", result.usage())
            if update_callback: await update_callback.send_analysis_update(result.data)
        except Exception as e:
            logger.error(f"Analysis agent '{name}' failed: {e}", exc_info=True)
            analysis_outputs[name] = None
    
    # 3. Final Report Generation
    logger.info(f"All analyses complete. Generating final report for {ticker}...")
    if update_callback: await update_callback.report_generation_start()

    report_generator_prompt = agents.REPORT_GENERATOR_USER_TEMPLATE.format(
        ticker=ticker,
        report_flavor=request.report_flavor,
        fundamental_analysis=agents.format_data_for_prompt(analysis_outputs.get("financials") or "Not available."),
        sentiment_analysis=agents.format_data_for_prompt(analysis_outputs.get("sentiment") or "Not available."),
        news_analysis=agents.format_data_for_prompt(analysis_outputs.get("news") or "Not available."),
        earnings_call_analysis=agents.format_data_for_prompt(analysis_outputs.get("earnings") or "Not available.")
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

async def run_financial_analysis_orchestration_wrapper(
    request: schemas.FinancialAnalysisRequest,
    agents_collection: agents.AgencyAgents,
    config: schemas.FinancialResearchConfig,
    services: dict,
    update_callback: None,
) -> schemas.FinancialResearchResponse:
    """
    Top-level wrapper for the orchestration to handle exceptions gracefully.
    """
    try:
        return await run_financial_analysis_orchestration(
            request=request,
            agents_collection=agents_collection,
            config=config,
            services=services,
            update_callback=update_callback
        )
    except Exception as e:
        logger.critical(f"CRITICAL UNHANDLED ERROR during financial orchestration: {e}", exc_info=True)
        if update_callback:
            await update_callback.send_error(e)
        # Return an error response
        return schemas.FinancialResearchResponse(
            final_report=schemas.FinancialReport(
                title=f"Error analyzing {request.stock_symbol}",
                executive_summary=f"A critical error occurred: {e}",
                fundamental_analysis_section="Not available due to error.",
                sentiment_analysis_section="Not available due to error.",
                news_analysis_section="Not available due to error.",
                earnings_call_section="Not available due to error.",
                final_conclusion="The analysis could not be completed."
            ),
            usage_statistics=UsageStatistics() # Return empty usage stats
        ) 