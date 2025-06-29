import os
import io
import base64
import numpy as np
import soundfile as sf
import torch
import whisper
import torchaudio
from pydantic import BaseModel
from typing import Optional

class STTRequest(BaseModel):
    audio: str          # Audio encodé en base64
    language: str = "fr"
    model: str = "base"

class STTResponse(BaseModel):
    text: str
    language: str
    confidence: float

async def transcribe_audio(
    audio_base64: str,
    language: str = "fr",
    model_name: str = "base"
) -> STTResponse:
    # --- décodage base64 ---
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    # --- lecture de l'audio ---
    audio_array, sample_rate = sf.read(audio_buffer)          # float64 par défaut
    audio_array = audio_array.astype(np.float32)              # → float32 obligatoire

    # passage en mono si stéréo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # (facultatif) resample à 16 kHz si besoin
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = torch.from_numpy(audio_array)
        audio_array = resampler(audio_tensor).numpy()
        sample_rate = 16000

    # --- chargement du modèle ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    # --- transcription ---
    result = model.transcribe(
        audio_array,
        language=language,
        task="transcribe",
        fp16=False                      # désactive le fp16 si ta carte ne le supporte pas
    )

    return STTResponse(
        text=result["text"],
        language=result["language"],
        confidence=result["segments"][0]["avg_logprob"] if result["segments"] else 0.0
    )