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
    audio: str  # audio encodé en base64
    language: str
    model: str

class STTResponse(BaseModel):
    text: str

async def transcribe_audio(audio_base64: str, language: str, model_name: str) -> STTResponse:
    # Implémente ici la logique de transcription
    return STTResponse(text="...texte reconnu...") 