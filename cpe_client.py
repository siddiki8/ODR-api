import asyncio
import websockets
import json
import argparse
import sys
from datetime import datetime
from typing import Optional

# Default server address for CPE
DEFAULT_WS_URL = "ws://localhost:8000/cpe/ws/cpe" # Match server route including prefix

def print_update(data: dict):
    """Formats and prints the received CPE status update."""
    now = datetime.now().strftime("%H:%M:%S")
    step = data.get('step', 'N/A')
    status = data.get('status', 'N/A')
    message = data.get('message', '')
    details = data.get('details')

    print(f"[{now}] [{step:<12}] [{status:<15}] - {message}", end='')

    if details:
        # Print simple key-value pairs nicely
        detail_items = []

        # Extract final results and usage based on CPE structure
        # Final profiles are not sent in the details of COMPLETE, they are saved in Firestore.
        # We primarily look for the count and usage stats.
        profiles_extracted_count = details.get('profiles_extracted') if step == 'COMPLETE' and status == 'END' else None
        usage_stats = details.get('usage') if step == 'COMPLETE' and status == 'END' else None

        # Process other details (excluding the ones handled above)
        other_details = {k: v for k, v in details.items() if k not in ['profiles_extracted', 'usage']}
        if other_details:
            try:
                details_str = json.dumps(other_details, indent=2)
                indented_details = "\n      " + details_str.replace("\n", "\n      ")
                detail_items.append(f"Details:{indented_details}")
            except Exception:
                detail_items.append(f"Details: {details}") # Fallback

        if detail_items:
            print(f"\n      {' '.join(detail_items)}")
        else:
             print() # Just a newline if only message was present
             
        # Print final count and usage stats separately
        if profiles_extracted_count is not None:
            print(f"\n--- Profiles Extracted: {profiles_extracted_count} ---")

        if usage_stats:
             print("\n--- Usage Statistics ---")
             try:
                 usage_str = json.dumps(usage_stats, indent=2)
                 print(usage_str)
             except Exception:
                 print(usage_stats) # Fallback
             print("------------------------")

    else:
        print() # Just add a newline if no details


async def run_client(uri: str, query: str, location: Optional[str], max_tasks: Optional[int]):
    """Connects to the CPE WebSocket server, sends query, and prints updates."""
    print(f"Connecting to {uri}...")
    try:
        # Include an Origin header so the server's CORS middleware allows the WebSocket upgrade
        async with websockets.connect(uri, origin="http://localhost:8000") as websocket:
            print(f"Connected! Sending query: '{query}'")
            if location: print(f"           Location: '{location}'")
            if max_tasks is not None: print(f"           Max Tasks: {max_tasks}")

            # Prepare the request payload (CPERequest format)
            request_payload = {
                "query": query,
                # Only include optional fields if they are provided
                **({"location": location} if location else {}),
                **({"max_search_tasks": max_tasks} if max_tasks is not None else {}),
            }
            
            await websocket.send(json.dumps(request_payload))
            print("Request sent. Waiting for updates...")

            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        print_update(data)
                        # Check for final messages
                        step = data.get('step', '').upper()
                        status = data.get('status', '').upper()
                        if step == 'COMPLETE' or status in ['FATAL', 'HANDLER_ERROR', 'VALIDATION_ERROR', 'JSON_ERROR', 'PROCESSING_ERROR', 'INTERNAL_ERROR']:
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
    parser = argparse.ArgumentParser(description="WebSocket client for CPE API.")
    parser.add_argument("query", help="The company/profile query to send.")
    parser.add_argument("-l", "--location", help="Optional location context for the query.")
    parser.add_argument("-m", "--max-tasks", type=int, help="Optional max search tasks for the planner.")
    parser.add_argument("-u", "--uri", default=DEFAULT_WS_URL,
                        help=f"WebSocket URI of the server (default: {DEFAULT_WS_URL})")

    args = parser.parse_args()

    if not args.query:
        print("Error: Query cannot be empty.")
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(run_client(args.uri, args.query, args.location, args.max_tasks))
    except KeyboardInterrupt:
        print("\nClient interrupted by user. Exiting.")
        sys.exit(0) 