import os
import torch
import whisper
from typing import Optional
from pydantic import BaseModel
import base64
import io
import numpy as np
import soundfile as sf

class STTRequest(BaseModel):
    audio: str  # base64
    language: str = "fr"
    model: str = "base"

class STTResponse(BaseModel):
    text: str
    language: str
    confidence: float

async def transcribe_audio(audio_base64: str, language: str = "fr", model_name: str = "base") -> STTResponse:
    """Transcrit l'audio en texte avec Whisper."""
    
    # Décodage de l'audio base64
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)
    
    # Chargement de l'audio
    audio_array, sample_rate = sf.read(audio_buffer)
    
    # Conversion en mono si stéréo
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Chargement du modèle Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    
    # Transcription
    result = model.transcribe(
        audio_array,
        language=language,
        task="transcribe"
    )
    
    return STTResponse(
        text=result["text"],
        language=result["language"],
        confidence=result["segments"][0]["confidence"] if result["segments"] else 0.0
    ) 