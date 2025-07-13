#!/bin/bash
set -e

echo "🚀 Pré-téléchargement des modèles pour optimiser le cache..."

# Créer les répertoires de cache
mkdir -p ~/.cache/tts
mkdir -p ~/.cache/huggingface
mkdir -p ~/.cache/ollama

# Pré-téléchargement des modèles TTS
echo "📥 Téléchargement des modèles TTS..."
python3 -c "
import os
from TTS.utils.manage import ModelManager

MODELS = [
    'facebook/mms-tts-fra',
    'tts_models/fr/css10/vits', 
    'tts_models/multilingual/multi-dataset/xtts_v2'
]

mm = ModelManager()
for model in MODELS:
    try:
        print(f'Téléchargement de {model}...')
        if model.startswith('facebook/'):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model, cache_dir=os.path.expanduser('~/.cache/tts'))
        else:
            mm.download_model(model)
        print(f'✅ {model} téléchargé')
    except Exception as e:
        print(f'⚠️  Échec pour {model}: {e}')
"

# Pré-téléchargement du modèle Ollama
echo "🤖 Téléchargement du modèle Mistral..."
ollama pull mistral || echo "⚠️  Échec du téléchargement Mistral"

echo "✅ Pré-téléchargement terminé !"
echo "📁 Cache disponible dans :"
echo "   - TTS: ~/.cache/tts"
echo "   - HuggingFace: ~/.cache/huggingface" 
echo "   - Ollama: ~/.cache/ollama" 