from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import asyncio
from agent import process_query
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PartSelect Agent API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    products: List[Dict] = []
    query_type: str = ""

@app.get("/")
async def root():
    return {
        "message": "PartSelect Agent API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PartSelect Agent"
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Streaming chat endpoint with SSE"""
    
    async def event_stream():
        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Process query through agent
            logger.info(f"Processing: {request.message}")
            result = await process_query(request.message, request.history)
            
            # Stream response in chunks
            response_text = result['response']
            
            chunk_size = 220
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                if chunk:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.05)
            
            # Send products
            if result['products']:
                # Convert any Decimal to float for JSON serialization
                products_serializable = []
                for product in result['products']:
                    prod_copy = dict(product)
                    if 'price' in prod_copy and prod_copy['price'] is not None:
                        prod_copy['price'] = float(prod_copy['price'])
                    products_serializable.append(prod_copy)
                
                yield f"data: {json.dumps({'type': 'products', 'products': products_serializable})}\n\n"
            
            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'query_type': result.get('query_type', '')})}\n\n"
            
            # Send done
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/reset")
async def reset_conversation():
    """Reset conversation history"""
    return {
        "message": "Conversation reset",
        "intro": "Hi! I'm your PartSelect assistant. I can help you with refrigerator and dishwasher parts - installation guides, compatibility checks, and troubleshooting. What can I help you with today?"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)