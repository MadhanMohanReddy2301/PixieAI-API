import os
import logging
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.agents import ChatHistoryAgentThread

from agents.pixie_agent.agent import PixieAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def build_thread_from_history(chat_history):
    chat = ChatHistory()
    for message in chat_history:
        role_str = message["role"].lower()
        try:
            role = AuthorRole(role_str)
        except ValueError:
            raise ValueError(f"Invalid role: {role_str}")
        chat_message = ChatMessageContent(role=role, content=message["content"])
        chat.add_message(chat_message)

    thread = ChatHistoryAgentThread(chat_history=chat)
    return thread


# Define the bot's behavior
class BusinessLogicBot():
    def __init__(self):
        super().__init__()
        self.response_text = None
        self.bernard_agent_instance = PixieAgent()
        self.agent = None

    async def _ensure_agent_initialized(self):
        if self.agent is None:
            self.agent = await self.bernard_agent_instance.get_agent()
        return self.agent

bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    bot = BusinessLogicBot()
    await bot._ensure_agent_initialized()
    logging.info("Bot initialized successfully")
    yield
    logging.info("Application shutting down...")


# FastAPI App
app = FastAPI(
    title="PixieAi",
    description="Microsoft Bot Framework with FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy", "service": "Bernard Bot"})


@app.post("/api/direct")
async def direct_message(request: Request):
    start_time = time.time()
    try:
        body = await request.json()
        user_message = body.get('message', '').strip()
        chat_history = body.get('chat_history', [])

        if not user_message:
            return JSONResponse({"error": "Message is required"}, status_code=400)

        thread = build_thread_from_history(chat_history)
        assistant_response = ""

        bernard_agent_instance = PixieAgent()
        agent = await bernard_agent_instance.get_agent()

        async for response in agent.invoke(messages=user_message, thread=thread):
            assistant_response = response.content
            result = response

        # input_tokens, output_tokens, total_tokens = bernard_agent_instance.extract_token_counts(result)
        execution_time = time.time() - start_time

        return JSONResponse({
            "success": True,
            "response": str(assistant_response),
            # "input_tokens": input_tokens,
            # "output_tokens": output_tokens,
            # "total_tokens": total_tokens,
            "response_time": round(execution_time, 3)
        })

    except Exception as e:
        logging.exception(f"Error in direct message: {e}")
        execution_time = time.time() - start_time
        return JSONResponse({
            "error": "Internal server error",
            "details": str(e),
            "execution_time_seconds": round(execution_time, 3)
        }, status_code=500)


if __name__ == "__main__":
    try:
        print("Starting Bernard Bot server...")
        print("Bot Framework endpoint: http://localhost:8000/api/messages")
        print("Direct message endpoint: http://localhost:8000/api/direct")
        print("Health check: http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.exception(f"Error starting app: {e}")
