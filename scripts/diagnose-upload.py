#!/usr/bin/env python3
"""
Script de diagnostic pour les problèmes d'upload WAV
"""

import os
import sys
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path

def check_environment():
    """Vérifie l'environnement système"""
    print("🔍 Diagnostic de l'environnement...")
    
    # Vérification des répertoires
    voices_dir = Path(os.getenv("VOICES_DIR", "voices"))
    print(f"📁 Répertoire voix : {voices_dir}")
    print(f"   - Existe : {voices_dir.exists()}")
    print(f"   - Écriture : {os.access(voices_dir, os.W_OK) if voices_dir.exists() else 'N/A'}")
    
    # Création du répertoire si nécessaire
    try:
        voices_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Répertoire créé/accessible")
    except Exception as e:
        print(f"   ❌ Erreur création : {e}")
    
    # Vérification de l'espace disque
    try:
        import shutil
        total, used, free = shutil.disk_usage(voices_dir)
        print(f"💾 Espace disque :")
        print(f"   - Total : {total // (1024**3)} GB")
        print(f"   - Utilisé : {used // (1024**3)} GB")
        print(f"   - Libre : {free // (1024**3)} GB")
    except Exception as e:
        print(f"   ❌ Erreur vérification espace : {e}")

def test_soundfile():
    """Teste la bibliothèque soundfile"""
    print("\n🎵 Test de soundfile...")
    
    try:
        # Création d'un fichier WAV de test
        sample_rate = 16000
        duration = 1.0  # 1 seconde
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        # Test d'écriture
        test_file = Path("test_audio.wav")
        sf.write(test_file, audio_data, sample_rate)
        print(f"   ✅ Écriture WAV : {test_file}")
        
        # Test de lecture
        data, sr = sf.read(test_file)
        print(f"   ✅ Lecture WAV : {len(data)} samples, {sr}Hz")
        
        # Nettoyage
        test_file.unlink()
        print(f"   ✅ Fichier de test supprimé")
        
    except Exception as e:
        print(f"   ❌ Erreur soundfile : {e}")

def test_torchaudio():
    """Teste torchaudio pour le resampling"""
    print("\n🎚️ Test de torchaudio...")
    
    try:
        import torch
        import torchaudio
        
        print(f"   ✅ PyTorch version : {torch.__version__}")
        print(f"   ✅ TorchAudio version : {torchaudio.__version__}")
        
        # Test de resampling
        sample_rate = 44100
        duration = 0.1
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        # Conversion en tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Resampling
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        resampled = resampler(audio_tensor)
        
        print(f"   ✅ Resampling : {sample_rate}Hz -> 16000Hz")
        print(f"   ✅ Résultat : {resampled.shape}")
        
    except Exception as e:
        print(f"   ❌ Erreur torchaudio : {e}")

def test_voice_service():
    """Teste le service de voix"""
    print("\n🎤 Test du service de voix...")
    
    try:
        from voice_service import save_voice_wav_file, list_voices, delete_voice
        
        # Création d'un fichier WAV de test
        sample_rate = 44100
        duration = 0.5
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        # Conversion en bytes
        import io
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        
        print(f"   ✅ Fichier de test créé : {len(audio_bytes)} bytes")
        
        # Test de sauvegarde
        voice_id = save_voice_wav_file(audio_bytes, "test_voice")
        print(f"   ✅ Voix sauvegardée : {voice_id}")
        
        # Test de liste
        voices = list_voices()
        print(f"   ✅ Voix disponibles : {voices}")
        
        # Test de suppression
        delete_voice(voice_id)
        print(f"   ✅ Voix supprimée")
        
    except Exception as e:
        print(f"   ❌ Erreur service voix : {e}")
        import traceback
        print(f"   📋 Traceback : {traceback.format_exc()}")

def main():
    """Fonction principale"""
    print("🚀 Diagnostic des problèmes d'upload WAV")
    print("=" * 50)
    
    check_environment()
    test_soundfile()
    test_torchaudio()
    test_voice_service()
    
    print("\n✅ Diagnostic terminé")

if __name__ == "__main__":
    main() 