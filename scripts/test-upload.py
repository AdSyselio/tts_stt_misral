#!/usr/bin/env python3
"""
Script de test pour l'upload WAV
"""

import requests
import json
import os
import sys
from pathlib import Path

def create_test_wav():
    """Crée un fichier WAV de test"""
    import numpy as np
    import soundfile as sf
    
    # Génération d'un son de test
    sample_rate = 16000
    duration = 2.0  # 2 secondes
    samples = int(sample_rate * duration)
    
    # Mélange de fréquences pour un son plus naturel
    t = np.linspace(0, duration, samples)
    audio_data = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # La 440Hz
        0.3 * np.sin(2 * np.pi * 880 * t) +  # La 880Hz
        0.2 * np.sin(2 * np.pi * 220 * t)    # La 220Hz
    )
    
    # Normalisation
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    
    # Sauvegarde
    test_file = Path("test_upload.wav")
    sf.write(test_file, audio_data, sample_rate)
    
    print(f"✅ Fichier de test créé : {test_file}")
    print(f"   - Taille : {test_file.stat().st_size} bytes")
    print(f"   - Durée : {duration}s")
    print(f"   - Sample rate : {sample_rate}Hz")
    
    return test_file

def test_upload(base_url, auth_token, file_path):
    """Teste l'upload d'un fichier WAV"""
    print(f"\n🚀 Test d'upload vers {base_url}")
    
    # Préparation de la requête
    url = f"{base_url}/voices"
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }
    
    # Préparation du fichier
    with open(file_path, "rb") as f:
        files = {
            "file": (file_path.name, f, "audio/wav")
        }
        data = {
            "name": "test_voice_upload"
        }
        
        print(f"📤 Upload en cours...")
        try:
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            
            print(f"📊 Réponse : {response.status_code}")
            print(f"📋 Headers : {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload réussi : {result}")
                return True
            else:
                print(f"❌ Erreur : {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout - L'upload prend trop de temps")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur réseau : {e}")
            return False

def test_health(base_url):
    """Teste la santé de l'API"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ API en ligne : {response.json()}")
            return True
        else:
            print(f"❌ API hors ligne : {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion API : {e}")
        return False

def main():
    """Fonction principale"""
    print("🧪 Test d'upload WAV")
    print("=" * 40)
    
    # Configuration
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    auth_token = os.getenv("AUTH_TOKEN", "your-token-here")
    
    print(f"🌐 URL de base : {base_url}")
    print(f"🔑 Token : {'*' * len(auth_token) if auth_token != 'your-token-here' else 'Non défini'}")
    
    # Test de santé
    if not test_health(base_url):
        print("❌ API non accessible, arrêt du test")
        return
    
    # Création du fichier de test
    test_file = create_test_wav()
    
    try:
        # Test d'upload
        success = test_upload(base_url, auth_token, test_file)
        
        if success:
            print("\n✅ Test d'upload réussi !")
        else:
            print("\n❌ Test d'upload échoué")
            
    finally:
        # Nettoyage
        if test_file.exists():
            test_file.unlink()
            print(f"🧹 Fichier de test supprimé")

if __name__ == "__main__":
    main() 