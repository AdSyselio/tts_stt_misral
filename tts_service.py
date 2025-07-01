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
    model: str = "siwis"   # "siwis", "css10" (ou "mai" pour compat.)
    voice_id: Optional[str] = None
    speed: float = 1.0
    

class TTSResponse(BaseModel):
    audio: str  # base64
    format: str = "wav"
    duration: float

async def synthesize_text(text: str, language: str = "fr", model: str = "siwis", voice_id: Optional[str] = None, speed: float = 1.0) -> TTSResponse:
    """Synthétise du texte en audio avec Coqui TTS."""
    
    # Correspondance des codes simples -> noms de modèles Coqui TTS
    param_map = {
        "siwis": "tts_models/fr/siwis/vits",   # voix féminine neutre (accent suisse-français)
        "css10": "tts_models/fr/css10/vits",  # voix féminine adulte
        "mai": "tts_models/fr/siwis/vits"     # alias pour compatibilité (l'ancien modèle mai/vits est indisponible)
    }

    # Priorité : paramètre explicite > variable d'environnement > défaut
    model_name_env = param_map.get(model, os.getenv("TTS_MODEL_NAME", "tts_models/fr/siwis/vits"))

    # Mise en cache d'une instance par process afin d'éviter un rechargement coûteux
    global _TTS_INSTANCE  # type: ignore
    if "_TTS_INSTANCE" not in globals() or getattr(_TTS_INSTANCE, "model_name", None) != model_name_env:
        try:
            _TTS_INSTANCE = TTS(model_name=model_name_env, gpu=torch.cuda.is_available())
        except Exception as err:
            # secours : revenir au modèle CSS10 si le modèle principal échoue (souvent pb de réseau)
            fallback = "tts_models/fr/css10/vits"
            _TTS_INSTANCE = TTS(model_name=fallback, gpu=torch.cuda.is_available())

    tts = _TTS_INSTANCE
    
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