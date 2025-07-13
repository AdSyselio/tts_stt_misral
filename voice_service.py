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


def _decode_and_save(audio_b64: str, voice_id: str) -> str:
    """Décodage commun et enregistrement du WAV dans VOICES_DIR."""
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


def save_voice_sample(audio_b64: str, name: str | None = None) -> str:
    """Enregistre un échantillon (base64) et renvoie l'ID de la voix.

    Args:
        audio_b64: Chaîne base64 de l'audio WAV/MP3.
        name: Identifiant facultatif (sinon UUID auto).
    """
    voice_id = name or uuid.uuid4().hex[:12]
    return _decode_and_save(audio_b64, voice_id)


def save_voice_sample_from_file(txt_path: str | Path, name: str | None = None) -> str:
    """Enregistre un échantillon audio à partir d'un fichier texte contenant la base64.

    Args:
        txt_path: Chemin du fichier texte (.txt) contenant la chaîne base64 (avec ou sans '\n').
        name: Identifiant facultatif ; sinon généré.

    Returns:
        L'identifiant de la voix sauvegardée.
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {txt_path}")
    audio_b64 = txt_path.read_text(encoding="utf-8").replace("\n", "").strip()
    voice_id = name or uuid.uuid4().hex[:12]
    return _decode_and_save(audio_b64, voice_id)


def delete_voice(voice_id: str) -> None:
    path = _voice_path(voice_id)
    if path.exists():
        path.unlink()


def save_voice_wav_file(content_bytes: bytes, name: str | None = None) -> str:
    """Enregistre un échantillon audio à partir d'un fichier WAV brut (bytes).

    Args:
        content_bytes: Contenu binaire du fichier WAV.
        name: Identifiant facultatif (sinon UUID auto).
    Returns:
        L'identifiant de la voix sauvegardée.
    """
    import io
    import uuid
    import soundfile as sf
    import os
    
    voice_id = name or uuid.uuid4().hex[:12]
    print(f"[PROCESS] Début traitement voix : voice_id={voice_id}")
    
    try:
        # Validation des bytes
        if not content_bytes or len(content_bytes) == 0:
            raise ValueError("Contenu audio vide")
        
        print(f"[PROCESS] Validation du format WAV...")
        wav_bytes = content_bytes
        
        # Test de lecture avec soundfile pour valider le format
        try:
            data, sr = sf.read(io.BytesIO(wav_bytes))
            print(f"[PROCESS] Audio lu : {len(data)} samples, {sr}Hz")
        except Exception as e:
            raise ValueError(f"Format audio invalide : {str(e)}")
        
        # Validation des paramètres audio
        if len(data) == 0:
            raise ValueError("Fichier audio vide")
        
        if sr <= 0:
            raise ValueError(f"Sample rate invalide : {sr}")
        
        # Resampling si nécessaire
        if sr != 16000:
            print(f"[PROCESS] Resampling {sr}Hz -> 16000Hz pour voice_id={voice_id}")
            try:
                import torchaudio, torch
                res = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                data = res(torch.from_numpy(data).float()).numpy()
                sr = 16000
                print(f"[PROCESS] Resampling terminé")
            except Exception as e:
                print(f"[PROCESS] Erreur resampling : {e}")
                raise ValueError(f"Erreur lors du resampling : {str(e)}")
        
        # Sauvegarde du fichier
        output_path = _voice_path(voice_id)
        print(f"[PROCESS] Sauvegarde vers : {output_path}")
        
        # Création du répertoire si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Écriture du fichier
        sf.write(output_path, data, sr)
        
        # Vérification que le fichier a bien été créé
        if not output_path.exists():
            raise RuntimeError("Le fichier n'a pas été créé")
        
        file_size = output_path.stat().st_size
        print(f"[PROCESS] Fichier sauvegardé : {file_size} bytes")
        
        print(f"[PROCESS] Fin traitement voix : voice_id={voice_id}")
        return voice_id
        
    except Exception as e:
        print(f"[PROCESS] Erreur dans save_voice_wav_file : {e}")
        # Nettoyage en cas d'erreur
        try:
            if 'output_path' in locals() and output_path.exists():
                output_path.unlink()
                print(f"[PROCESS] Fichier temporaire supprimé")
        except:
            pass
        raise e 