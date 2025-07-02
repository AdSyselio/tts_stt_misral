from fastapi import FastAPI, HTTPException, Depends, Header
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
from voice_service import save_voice_sample, delete_voice, list_voices
import uuid

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

def _ollama_base() -> str:
    """Renvoie l'URL de base Ollama avec schéma http:// si nécessaire."""
    host = os.getenv("OLLAMA_HOST", "ollama:11434")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")

async def get_ollama_response(
    messages: list[Message],
    temperature: float,
    max_tokens: int,
    model_name: str | None = None,
) -> str:
    """Appelle l'API Ollama et renvoie la réponse.

    Si *model_name* est fourni, on l'utilise ; sinon on retombe sur la variable
    d'environnement MODEL_NAME ou, à défaut, « mistral »."""

    ollama_host = _ollama_base()
    model_name = model_name or os.getenv("MODEL_NAME", "mistral")
    
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

# -----------------------------------------------------------------------------
# Route de compatibilité OpenAI (n8n AI Agent)
# -----------------------------------------------------------------------------

# L'AI Agent d'n8n s'attend à un endpoint POST /v1/chat/completions au format
# OpenAI. Nous l'implémentons en proxy vers get_ollama_response.

# Permettre l'auth par clé secrète RunPod (Bearer <SECRET_KEY> ou X-API-KEY)

@app.post("/v1/chat/completions", tags=["Compatibility"], include_in_schema=False)
async def openai_compat(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Compatibilité OpenAI v1 pour Ollama.

    Authentification :
    • JWT classique (Authorization: Bearer <token>)
    • Clé secrète RunPod (SECRET_KEY) passée :
        – soit dans l'en-tête `Authorization: Bearer <SECRET_KEY>`
        – soit dans l'en-tête `X-API-KEY: <SECRET_KEY>`
    """

    # --- Auth ----------------------------------------------------------------
    secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")

    provided_token = None
    if authorization and authorization.lower().startswith("bearer "):
        provided_token = authorization.split(" ", 1)[1]
    elif x_api_key:
        provided_token = x_api_key

    if provided_token != secret_key:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    try:
        messages_in = payload.get("messages", [])
        model = payload.get("model", os.getenv("MODEL_NAME", "mistral"))
        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens", 1024)

        # Conversion vers notre modèle Message
        msg_objs = [Message(role=m["role"], content=m["content"]) for m in messages_in]

        answer = await get_ollama_response(msg_objs, temperature, max_tokens, model)

        # Réponse au format OpenAI
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/chat/completions", include_in_schema=False)
async def openai_compat_get(
    model: str,
    prompt: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """Fallback GET pour anciennes versions n8n (Ollama Chat Model)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return await openai_compat(payload, authorization, x_api_key)

# -----------------------------------------------------------------------------
# Endpoint /v1/models  (utilisé par n8n pour tester la connexion OpenAI)
# -----------------------------------------------------------------------------

@app.get("/v1/models", tags=["Compatibility"], include_in_schema=False)
async def list_models(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Renvoie la liste des modèles disponibles au format OpenAI.

    Nécessaire pour que le test de connexion OpenAI (n8n) passe.
    """

    secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1]
    elif x_api_key:
        provided = x_api_key

    if provided != secret_key:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    model_name = os.getenv("MODEL_NAME", "mistral")
    return {
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": "runpod-local"
            }
        ],
        "object": "list"
    }

# -- Variant with trailing slash ------------------------------------------------

@app.get("/v1/models/", include_in_schema=False)
async def list_models_slash(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    """Alias avec slash final pour compatibilité clients."""
    return await list_models(authorization, x_api_key)

# Alias pour /v1/chat/completions/ (POST et GET) --------------------------------

@app.post("/v1/chat/completions/", include_in_schema=False)
async def openai_compat_post_slash(payload: Dict[str, Any], authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    return await openai_compat(payload, authorization, x_api_key)

@app.get("/v1/chat/completions/", include_in_schema=False)
async def openai_compat_get_slash(
    model: str,
    prompt: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    return await openai_compat_get(model, prompt, authorization, x_api_key, temperature, max_tokens)

# -----------------------------------------------------------------------------
# Gestion des voix clonées (XTTS)
# -----------------------------------------------------------------------------

class VoiceUploadRequest(BaseModel):
    name: Optional[str] = None       # identifiant souhaité (optionnel)
    audio: str                       # wav ou mp3 en base64 (≥ 5 secondes recommandées)

class VoiceUploadResponse(BaseModel):
    voice_id: str

@app.post("/voices", tags=["Voices"], response_model=VoiceUploadResponse)
async def upload_voice(req: VoiceUploadRequest, current_user: TokenData = Depends(get_current_user)):
    """Upload et enregistre un échantillon pour XTTS."""
    vid = save_voice_sample(req.audio, req.name)
    return VoiceUploadResponse(voice_id=vid)

@app.get("/voices", tags=["Voices"], response_model=list)
async def list_available_voices(current_user: TokenData = Depends(get_current_user)):
    """Liste des voix clonées disponibles."""
    return list_voices()

@app.delete("/voices/{voice_id}", tags=["Voices"])
async def remove_voice(voice_id: str, current_user: TokenData = Depends(get_current_user)):
    """Supprime un échantillon de voix."""
    delete_voice(voice_id)
    return {"status": "deleted", "voice_id": voice_id}

# -----------------------------------------------------------------------------
# Route de compatibilité Ollama native (/api/chat)
# -----------------------------------------------------------------------------

@app.post("/api/chat", tags=["Compatibility"], include_in_schema=False)
async def ollama_native_chat(payload: Dict[str, Any]):
    """Compatibilité avec l'endpoint natif d'Ollama (/api/chat).

    Cette route attend un JSON conforme à l'API Ollama :
    {
        "model": "mistral",
        "messages": [...],
        "options": {"temperature": 0.7, "num_predict": 1024}
    }
    Elle renvoie la réponse dans le même format qu'Ollama afin que les
    intégrations prévues (par ex. le n8n « Ollama Chat Model ») fonctionnent
    sans modification côté client.
    """
    # --- Extraction des paramètres ---
    try:
        msgs_in = payload.get("messages", [])
        if not isinstance(msgs_in, list):
            raise ValueError("'messages' doit être une liste")

        temperature = (
            payload.get("options", {}).get("temperature")
            or payload.get("temperature", 0.7)
        )
        max_tokens = (
            payload.get("options", {}).get("num_predict")
            or payload.get("max_tokens", 1024)
        )

        messages = [Message(role=m["role"], content=m["content"]) for m in msgs_in]
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

    # --- Appel du backend Ollama ---
    model_req = payload.get("model")
    answer = await get_ollama_response(messages, temperature, max_tokens, model_req)

    # --- Réponse conforme Ollama ---
    return {
        "model": model_req or os.getenv("MODEL_NAME", "mistral"),
        "created_at": datetime.utcnow().isoformat(),
        "message": {
            "role": "assistant",
            "content": answer,
        },
        "done": True,
    }

# -----------------------------------------------------------------------------
# Route de compatibilité Ollama pour la liste des modèles (/api/tags)
# -----------------------------------------------------------------------------

@app.get("/api/tags", tags=["Compatibility"], include_in_schema=False)
async def ollama_native_tags():
    """Renvoie la liste des modèles disponibles depuis l'instance Ollama.

    Cette route est appelée par les clients comme n8n pour remplir le menu
    déroulant des modèles.
    """
    ollama_host = _ollama_base()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{ollama_host}/api/tags", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as err:
            raise HTTPException(status_code=500, detail=f"Erreur Ollama: {err}")