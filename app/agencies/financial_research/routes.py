from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
from app.core.dependencies import get_llm_provider
from .orchestrator import run_financial_analysis_orchestration
from .agents import AgencyAgents
from . import schemas
from .callbacks import FinancialsWebSocketUpdateHandler

# Import services
from app.services.finance_services.yfinance_service import YFinanceService
from app.services.finance_services.social_media_service import MockSocialMediaService
from app.services.finance_services.news_service import MockNewsService
from app.services.finance_services.earnings_call_service import MockEarningsCallService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/analyze")
async def analysis_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Instantiate core components
    llm_provider = get_llm_provider() # Assumes a default provider setup
    
    # Instantiate services
    services = {
        "yfinance_service": YFinanceService(),
        "social_service": MockSocialMediaService(),
        "news_service": MockNewsService(),
        "earnings_service": MockEarningsCallService(),
    }
    
    try:
        # Receive user request
        request_data = await websocket.receive_json()
        request = schemas.FinancialAnalysisRequest.model_validate(request_data)

        # Setup agency-specific components
        agency_config = schemas.FinancialResearchConfig()
        # You could potentially merge request.config_overrides here
        
        agents_collection = AgencyAgents(llm_provider=llm_provider, config=agency_config)
        
        # Attach config to agents_collection for easy access in helpers
        agents_collection.config = agency_config

        # Setup callback handler
        async def send_update(payload: dict):
            await websocket.send_json(payload)
        
        update_handler = FinancialsWebSocketUpdateHandler(send_update)

        # Run the orchestration
        response = await run_financial_analysis_orchestration(
            request=request,
            agents_collection=agents_collection,
            services=services,
            update_callback=update_handler 
        )

        # Send the final report via the handler
        await update_handler.send_final_report(response.final_report)

    except WebSocketDisconnect:
        logger.info("Client disconnected from financial analysis websocket.")
    except Exception as e:
        logger.error(f"Error in Financial Research WebSocket: {e}", exc_info=True)
        # Inform the client of the error
        if 'update_handler' in locals():
            await update_handler.send_error(e)
    finally:
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.close() 