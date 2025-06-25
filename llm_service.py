import httpx
import os
from typing import List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

async def get_ollama_response(messages: List[Message], temperature: float, max_tokens: int) -> str:
    """Appelle l'API Ollama pour obtenir une réponse du modèle."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model_name = os.getenv("MODEL_NAME", "mistral")
    
    async with httpx.AsyncClient() as client:
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