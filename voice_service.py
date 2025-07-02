from pathlib import Path
import base64
import uuid
import io
import soundfile as sf
from typing import List, Dict
import os

VOICES_DIR = Path(os.getenv("VOICES_DIR", "voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _voice_path(voice_id: str) -> Path:
    return VOICES_DIR / f"{voice_id}.wav"


def list_voices() -> List[str]:
    """Retourne la liste des identifiants de voix disponibles."""
    return [p.stem for p in VOICES_DIR.glob("*.wav")]


def save_voice_sample(audio_b64: str, name: str | None = None) -> str:
    """Enregistre un échantillon (base64) et renvoie l'ID de la voix."""
    voice_id = name or uuid.uuid4().hex[:12]
    wav_bytes = base64.b64decode(audio_b64)
    data, sr = sf.read(io.BytesIO(wav_bytes))
    if sr != 16000:
        # resample simple fallback
        import torchaudio, torch
        res = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        data = res(torch.from_numpy(data).float()).numpy()
        sr = 16000
    sf.write(_voice_path(voice_id), data, sr)
    return voice_id


def delete_voice(voice_id: str) -> None:
    path = _voice_path(voice_id)
    if path.exists():
        path.unlink() 