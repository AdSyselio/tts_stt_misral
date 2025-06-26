from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import httpx
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
from fastapi.responses import Response
from auth import Token, authenticate_user, create_access_token, get_current_user, TokenData, ACCESS_TOKEN_EXPIRE_MINUTES
from tts_service import TTSRequest, TTSResponse, synthesize_text
from stt_service import STTRequest, STTResponse, transcribe_audio
from llm_service import Message, ChatRequest, get_ollama_response

# Métriques Prometheus
REQUESTS = Counter('http_requests_total', 'Total des requêtes HTTP', ['method', 'endpoint'])
LATENCY = Histogram('http_request_duration_seconds', 'Latence des requêtes HTTP', ['method', 'endpoint'])

app = FastAPI(
    title="IA Bot - Core API"
)

class ChatResponse(BaseModel):
    response: str
    timestamp: str

class HomeResponse(BaseModel):
    status: str
    version: str
    description: str

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

@app.get("/", response_model=HomeResponse)
async def home():
    """Page d'accueil Core"""
    return HomeResponse(
        status="online",
        version="1.0.0",
        description="Service principal LLM et orchestration pour RunPod"
    )

@app.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authentification"""
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=401,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/llm/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: TokenData = Depends(get_current_user)):
    """Chat avec Mistral"""
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

@app.post("/tts", response_model=TTSResponse, 
    tags=["Speech"],
    summary="Conversion texte vers parole",
    description="Convertit du texte en audio en utilisant le modèle TTS Coqui"
)
async def text_to_speech(request: TTSRequest, current_user: TokenData = Depends(get_current_user)):
    """
    Conversion texte vers parole avec les paramètres suivants:
    - **text**: Le texte à convertir en audio
    - **language**: La langue du texte (par défaut: fr)
    - **voice_id**: L'identifiant de la voix (optionnel)
    - **speed**: La vitesse de la parole (par défaut: 1.0)
    """
    try:
        return await synthesize_text(
            text=request.text,
            language=request.language,
            voice_id=request.voice_id,
            speed=request.speed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stt", response_model=STTResponse, 
    tags=["Speech"],
    summary="Conversion parole vers texte",
    description="Convertit un fichier audio en texte en utilisant Whisper"
)
async def speech_to_text(request: STTRequest, current_user: TokenData = Depends(get_current_user)):
    """
    Conversion parole vers texte avec les paramètres suivants:
    - **audio**: L'audio en base64
    - **language**: La langue de l'audio (par défaut: fr)
    - **model**: Le modèle Whisper à utiliser (par défaut: base)
    """
    try:
        return await transcribe_audio(
            audio_base64=request.audio,
            language=request.language,
            model_name=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Documentation Swagger UI"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="IA Bot - Core API - Documentation",
        swagger_favicon_url="/favicon.ico"
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Documentation ReDoc"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="IA Bot - Core API - Documentation",
        redoc_favicon_url="/favicon.ico"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI Schema"""
    return get_openapi(
        title="IA Bot - Core API",
        version="1.0.0",
        description="Service principal LLM et orchestration pour RunPod",
        routes=app.routes,
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Métriques Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Vérification de santé"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "llm": "ok",
            "tts": "ok",
            "stt": "ok"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 