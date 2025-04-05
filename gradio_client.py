import gradio as gr
import asyncio
import websockets
import json
import logging
from datetime import datetime

# Configure basic logging for the client function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - GRADIO_CLIENT - %(levelname)s - %(message)s')
logger = logging.getLogger("GradioClient")

# --- Configuration ---\
WEBSOCKET_URL = "ws://localhost:8000/ws/research" # Adjust if needed

async def run_research_gradio(query: str):
    """
    Connects to the WebSocket, runs research, and yields updates for Gradio.

    Yields:
        tuple[str, str]: A tuple containing (status_update_string, final_report_string)
    """
    status_log = ""
    final_report = "*(Report will appear here upon completion)*"

    # Initial yield to clear previous state
    yield status_log, final_report

    if not query:
        status_log += "[INFO] Please enter a research query.\n"
        yield status_log, final_report
        return

    status_log += f"[INFO] Connecting to WebSocket: {WEBSOCKET_URL}\n"
    yield status_log, final_report

    try:
        async with websockets.connect(WEBSOCKET_URL, open_timeout=10) as websocket:
            status_log += "[INFO] WebSocket connection established.\n"
            yield status_log, final_report

            # 1. Send the initial research request
            request_payload = {"query": query} # Add other params if needed
            request_payload_str = json.dumps(request_payload)
            status_log += f"[INFO] Sending research request for: \"{query[:50]}...\"\n"
            yield status_log, final_report
            await websocket.send(request_payload_str)
            status_log += "[INFO] Request sent. Waiting for updates...\n"
            yield status_log, final_report

            # 2. Listen for and yield status updates
            while True:
                message_str = await websocket.recv()
                try:
                    message_data = json.loads(message_str)
                    step = message_data.get("step", "UNKNOWN_STEP")
                    status = message_data.get("status", "UNKNOWN_STATUS")
                    message = message_data.get("message", "")
                    details = message_data.get("details", {})

                    ts = datetime.now().strftime("%H:%M:%S")
                    status_line = f"[{ts} | {step}/{status}] {message}"

                    # Add specific details if useful
                    if step == "PLANNING" and status == "END" and details.get("plan"):
                            plan = details["plan"]
                            queries = plan.get('search_queries', [])
                            status_line += f" (Plan: {plan.get('writing_plan',{}).get('overall_goal','?')}, Queries: {len(queries)})"
                    elif step == "SEARCHING" and status == "END":
                        status_line += f" (Results: {details.get('raw_result_count', '?')})"
                    elif step == "RANKING" and status == "END":
                        status_line += f" (Summarize: {details.get('num_summarize','?')}, Chunk: {details.get('num_chunk','?')})" # Hypothetical detail keys
                    elif step == "PROCESSING" and status == "INFO":
                            status_line += f" (Summarized: {details.get('summarized_count','?')})" # Hypothetical detail keys
                    elif step == "PROCESSING" and status == "END":
                            status_line += f" (Chunks Found: {details.get('relevant_chunk_count','?')})" # Hypothetical detail keys


                    status_log += status_line + "\n"
                    yield status_log, final_report # Update status log

                    # Check for completion or error
                    if step == "COMPLETE" and status == "END":
                        status_log += "[INFO] Received 'COMPLETE' message. Final report received.\n"
                        final_report = details.get("final_report", "*No report content found in final message.*")
                        yield status_log, final_report # Update both status and final report
                        break # Exit loop

                    elif step == "ERROR":
                            error_msg = f"[ERROR/{status}] {message} Details: {details}"
                            status_log += error_msg + "\n"
                            final_report = f"**Research Failed**\n\n{error_msg}"
                            yield status_log, final_report # Update status and report with error
                            break # Exit loop

                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message_str}")
                    status_log += f"[WARN] Received non-JSON message: {message_str[:100]}...\n"
                    yield status_log, final_report
                except Exception as e:
                    logger.error(f"Error processing received message: {e}", exc_info=True)
                    status_log += f"[ERROR] Error processing message: {e}\n"
                    yield status_log, final_report
                    break # Exit loop on processing error


    except websockets.exceptions.ConnectionClosedOK:
        status_log += "[INFO] Server closed the connection normally.\n"
    except websockets.exceptions.ConnectionClosedError as e:
        status_log += f"[ERROR] Server closed the connection with error: {e}\n"
        final_report = f"**Connection Error**\n\nServer closed connection unexpectedly: {e}"
    except ConnectionRefusedError:
        status_log += f"[ERROR] Connection refused. Is the FastAPI server running at {WEBSOCKET_URL}?\n"
        final_report = f"**Connection Error**\n\nConnection refused. Please ensure the API server is running."
    except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            status_log += f"[ERROR] An unexpected error occurred: {e}\n"
            final_report = f"**Client Error**\n\nAn unexpected error occurred in the Gradio client: {e}"

    yield status_log, final_report # Final update


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Deep Research API - Gradio Client")
    gr.Markdown("Enter a query below and click 'Run Research'. Status updates will stream in the log, and the final report will appear on the right.")

    with gr.Row():
        query_input = gr.Textbox(
            label="Research Query",
            placeholder="e.g., Analyze the efficacy of TDA vs deep learning for drug discovery...",
            lines=2
        )

    run_button = gr.Button("Run Research")

    with gr.Row():
        status_output = gr.Textbox(
            label="Status Log",
            lines=20,
            interactive=False,
            autoscroll=True,
            max_lines=40 # Limit max lines to prevent browser slowdown
        )
        report_output = gr.Markdown(
            label="Final Report",
            value="*(Report will appear here upon completion)*"
        )

    # Connect button click to the backend function
    run_button.click(
        fn=run_research_gradio,
        inputs=[query_input],
        outputs=[status_output, report_output]
    )

if __name__ == "__main__":
    demo.launch() # Share=False for local testing