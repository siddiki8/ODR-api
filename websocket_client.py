# websocket_client.py

import asyncio
import websockets
import json
import logging
import os
from datetime import datetime
import re # For sanitizing filenames

# Configure logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSocketClient")

# --- Configuration ---
# Adjust this URL if your FastAPI server runs on a different host or port
WEBSOCKET_URL = "ws://localhost:8000/ws/research"
# Sample research request payload
RESEARCH_PAYLOAD = {
    "query": "Compare the environmental and economic sustainability of various carbon capture technologies (direct air capture, bioenergy with carbon capture, enhanced weathering, and ocean-based methods), analyzing their scalability, cost-effectiveness, and potential ecological impacts based on peer-reviewed literature.",
    # Optional: Add other ResearchRequest fields here if needed for testing
    # e.g., max_search_tasks: 5, llm_provider: 'google'
    # "planner_llm_config": null,
    # "summarizer_llm_config": null,
    # "writer_llm_config": null,
}

async def run_research_client():
    """Connects to the WebSocket server, sends a request, and prints updates."""
    logger.info(f"Attempting to connect to WebSocket: {WEBSOCKET_URL}")
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            logger.info("WebSocket connection established.")

            # 1. Send the initial research request
            request_payload_str = json.dumps(RESEARCH_PAYLOAD)
            logger.info(f"Sending research request: {request_payload_str}")
            await websocket.send(request_payload_str)
            logger.info("Request sent.")

            # 2. Listen for and print status updates
            logger.info("Waiting for status updates from the server...")
            try:
                while True:
                    message_str = await websocket.recv()
                    try:
                        message_data = json.loads(message_str)
                        logger.info(f"Received update: {message_data}")
                        # Optional: Check for the 'COMPLETE' step to potentially exit early
                        if message_data.get("step") == "COMPLETE" and message_data.get("status") == "END":
                            logger.info("Received 'COMPLETE' message. Processing final report...")
                            # --- Save Final Report --- #
                            try:
                                report_content = message_data.get("details", {}).get("final_report")
                                if report_content:
                                    # Create a reports directory if it doesn't exist
                                    output_dir = "./research_reports"
                                    os.makedirs(output_dir, exist_ok=True)

                                    # Sanitize query for filename
                                    query_part = RESEARCH_PAYLOAD.get("query", "untitled")[:50] # Limit length
                                    sanitized_query = re.sub(r'[^a-zA-Z0-9_\-]', '_', query_part).strip('_')
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"{timestamp}_{sanitized_query}.md"
                                    filepath = os.path.join(output_dir, filename)

                                    with open(filepath, 'w', encoding='utf-8') as f:
                                        f.write(report_content)
                                    logger.info(f"Final report saved to: {filepath}")
                                else:
                                     logger.warning("'COMPLETE' message received, but no 'final_report' found in details.")
                            except Exception as save_e:
                                logger.error(f"Failed to save the final report: {save_e}", exc_info=True)
                            # ------------------------- #
                            break # Exit loop after processing COMPLETE
                        elif message_data.get("step") == "ERROR": # Check for general ERROR step
                             error_status = message_data.get("status", "ERROR") # e.g., ERROR, FATAL
                             logger.error(f"Received 'ERROR' message: {message_data.get('message', '')} Details: {message_data.get('details')}")
                             logger.warning("Closing connection due to server error.")
                             break

                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message: {message_str}")
                    except Exception as e:
                        logger.error(f"Error processing received message: {e}", exc_info=True)


            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Server closed the connection normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"Server closed the connection with error: {e}")
            except Exception as e:
                 logger.error(f"An unexpected error occurred while receiving messages: {e}", exc_info=True)

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {WEBSOCKET_URL}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at {WEBSOCKET_URL.replace('ws://', 'http://')}?")
    except Exception as e:
        logger.error(f"Failed to connect or an error occurred: {e}", exc_info=True)

    logger.info("Client finished.")

if __name__ == "__main__":
    asyncio.run(run_research_client()) 