import os
import torch
from TTS.api import TTS
from typing import Optional
from pydantic import BaseModel
import base64
import io
import soundfile as sf

class TTSRequest(BaseModel):
    text: str
    language: str = "fr"
    voice_id: Optional[str] = None
    speed: float = 1.0

class TTSResponse(BaseModel):
    audio: str  # base64
    format: str = "wav"
    duration: float

async def synthesize_text(text: str, language: str = "fr", voice_id: Optional[str] = None, speed: float = 1.0) -> TTSResponse:
    """Synthétise du texte en audio avec Coqui TTS."""
    
    # Initialisation de Coqui TTS
    tts = TTS(model_name="tts_models/fr/css10/vits", gpu=torch.cuda.is_available())
    
    # Buffer pour l'audio
    audio_buffer = io.BytesIO()
    
    # Génération audio
    wav = tts.tts(text=text, speaker=voice_id)
    
    # Ajustement de la vitesse si nécessaire
    if speed != 1.0:
        wav = tts.adjust_speed(wav, speed)
    
    # Sauvegarde dans le buffer (TTS 0.21.x n'expose plus save_wav)
    sf.write(audio_buffer, wav, tts.synthesizer.output_sample_rate, format='WAV')
    audio_buffer.seek(0)
    
    # Conversion en base64
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    
    return TTSResponse(
        audio=audio_base64,
        format="wav",
        duration=len(wav) / tts.synthesizer.output_sample_rate
    ) 