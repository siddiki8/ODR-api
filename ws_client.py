import asyncio
import websockets
import json
import argparse
import sys
from datetime import datetime

# Default server address
DEFAULT_WS_URL = "ws://localhost:8000/deep_research/ws/research"

def print_update(data: dict):
    """Formats and prints the received status update."""
    now = datetime.now().strftime("%H:%M:%S")
    step = data.get('step', 'N/A')
    status = data.get('status', 'N/A')
    message = data.get('message', '')
    details = data.get('details')

    print(f"[{now}] [{step:<12}] [{status:<15}] - {message}", end='')

    if details:
        # Print simple key-value pairs nicely
        detail_items = []

        # Extract final report and usage based on new step/status structure
        final_report = details.get('final_report') if step == 'FINALIZING' and status == 'END' else None
        usage_stats = details.get('usage') if step == 'COMPLETE' and status == 'END' else None

        # Process other details (excluding the ones handled above)
        other_details = {k: v for k, v in details.items() if k not in ['final_report', 'usage']}
        if other_details:
            try:
                # Try pretty printing non-result details JSON
                details_str = json.dumps(other_details, indent=2)
                # Indent the details block
                indented_details = "\n      " + details_str.replace("\n", "\n      ")
                detail_items.append(f"Details:{indented_details}")
            except Exception:
                detail_items.append(f"Details: {details}") # Fallback

        if detail_items:
            print(f"\n      {' '.join(detail_items)}")
        else:
             print() # Just a newline if only message was present

        # Print final report and usage stats separately for clarity
        if usage_stats:
             print("\n--- Usage Statistics ---")
             try:
                 # Pretty print usage stats JSON
                 usage_str = json.dumps(usage_stats, indent=2)
                 print(usage_str)
             except Exception:
                 print(usage_stats) # Fallback
             print("------------------------")

        if final_report:
            print("\n------ Final Report ------")
            print(final_report)
            print("------------------------")

    else:
        print() # Just add a newline if no details


async def run_client(uri: str, query: str):
    """Connects to the WebSocket server, sends query, and prints updates."""
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected! Sending query: '{query}'")

            # Prepare the request payload (CoreResearchRequest format)
            request_payload = {"query": query}
            await websocket.send(json.dumps(request_payload))
            print("Query sent. Waiting for updates...")

            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        print_update(data)
                        # Check for final messages to potentially exit cleanly
                        step = data.get('step', '').upper()
                        status = data.get('status', '').upper()
                        if step == 'COMPLETE' or status in ['FATAL', 'HANDLER_ERROR', 'VALIDATION_ERROR']:
                            print("\nReceived final status. Closing connection.")
                            break
                    except json.JSONDecodeError:
                        print(f"<Received non-JSON message: {message}>")
                    except Exception as e:
                        print(f"<Error processing message: {e}>")
                        print(f"<Raw message: {message}>")

            except websockets.exceptions.ConnectionClosedOK:
                print("\nConnection closed normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"\nConnection closed with error: {e}")

    except websockets.exceptions.InvalidURI:
        print(f"Error: Invalid WebSocket URI: {uri}")
        sys.exit(1)
    except ConnectionRefusedError:
         print(f"Error: Connection refused. Is the server running at {uri}?")
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket client for Deep Research API.")
    parser.add_argument("query", help="The research query to send to the server.")
    parser.add_argument("-u", "--uri", default=DEFAULT_WS_URL,
                        help=f"WebSocket URI of the server (default: {DEFAULT_WS_URL})")

    args = parser.parse_args()

    # Basic validation
    if not args.query:
        print("Error: Query cannot be empty.")
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(run_client(args.uri, args.query))
    except KeyboardInterrupt:
        print("\nClient interrupted by user. Exiting.")
        sys.exit(0) 