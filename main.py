from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import httpx
import os
from typing import Optional
from datetime import datetime

app = FastAPI(title="LLM Service")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    response: str
    timestamp: str

async def get_ollama_response(messages: list[Message], temperature: float, max_tokens: int) -> str:
    ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model_name = os.getenv("MODEL_NAME", "mistral")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Erreur Ollama: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_text = await get_ollama_response(
            request.messages,
            request.temperature,
            request.max_tokens
        )
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 