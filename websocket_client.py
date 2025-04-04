# websocket_client.py

import asyncio
import websockets
import json
import logging

# Configure logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSocketClient")

# --- Configuration ---
# Adjust this URL if your FastAPI server runs on a different host or port
WEBSOCKET_URL = "ws://localhost:8000/ws/research"
# Sample research request payload
RESEARCH_PAYLOAD = {
    "query": "What are the latest advancements in quantum computing?",
    # Optional: Add other ResearchRequest fields here if needed for testing
    # "planner_llm_config": null,
    # "summarizer_llm_config": null,
    # "writer_llm_config": null,
    # "scraper_strategies": null
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
                            logger.info("Received 'COMPLETE' message. Closing connection.")
                            break
                        elif message_data.get("step") == "ERROR" and message_data.get("status") == "ERROR":
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