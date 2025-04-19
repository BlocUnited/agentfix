from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Callable, Awaitable, List
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

initialization_coroutines: List[Callable[[], Awaitable[None]]] = []

def add_initialization_coroutine(coroutine: Callable[[], Awaitable[None]]):
    initialization_coroutines.append(coroutine)

@app.on_event("startup")
async def startup_event():
    for init_coroutine in initialization_coroutines:
        await init_coroutine()

# WebSocket Manager
class SharedWebSocketManager:
    def __init__(self):
        # Storing connections with their associated chat_id
        self.active_connections: Dict[str, WebSocket] = {}
        # Handlers for different chat types
        self.handlers: Dict[str, Callable[[WebSocket, str], Awaitable[None]]] = {}
        # Store user and enterprise data by chat_id
        self.connection_info: Dict[str, Dict[str, str]] = {}  # Storing user_id and enterprise_id per connection

    async def connect(self, websocket: WebSocket, chat_id: str, user_id: str, enterprise_id: str):
        # Accept the WebSocket connection
        await websocket.accept()
        # Track the WebSocket connection by chat_id
        self.active_connections[chat_id] = websocket
        # Store user and enterprise info
        self.connection_info[chat_id] = {
            "user_id": user_id,
            "enterprise_id": enterprise_id
        }
        logger.info(f"Connected: chat_id={chat_id}, user_id={user_id}, enterprise_id={enterprise_id}")

    async def disconnect(self, chat_id: str):
        # Remove the connection and associated info when disconnected
        self.active_connections.pop(chat_id, None)
        self.connection_info.pop(chat_id, None)
        logger.info(f"Disconnected: {chat_id}")

    def register_handler(self, chat_type: str, handler: Callable[[WebSocket, str, str, str], Awaitable[None]]):
        # Register a handler for a specific chat type
        self.handlers[chat_type] = handler

    async def handle_websocket(self, websocket: WebSocket, chat_id: str, chat_type: str, user_id: str, enterprise_id: str):
        # Establish the connection and log user/enterprise details
        await self.connect(websocket, chat_id, user_id, enterprise_id)
        try:
            if chat_type in self.handlers:
                # Pass user and enterprise info to the handler
                await self.handlers[chat_type](websocket, chat_id, user_id, enterprise_id)
            else:
                logger.error(f"No handler registered for chat_type: {chat_type}")
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {chat_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {str(e)}")
        finally:
            # Ensure disconnection is handled
            await self.disconnect(chat_id)

shared_websocket_manager = SharedWebSocketManager()

# WebSocket endpoint
@app.websocket("/ws/{chat_type}/{chat_id}/{user_id}/{enterprise_id}")
async def websocket_endpoint(websocket: WebSocket, chat_type: str, chat_id: str, user_id: str, enterprise_id: str):
    # Handle the WebSocket connection with provided user and enterprise IDs
    await shared_websocket_manager.handle_websocket(websocket, chat_id, chat_type, user_id, enterprise_id)

