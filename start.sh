#!/bin/bash
set -e

# 1. Démarrer Ollama en arrière-plan
echo "Lancement d'Ollama..."
ollama serve > /tmp/ollama.log 2>&1 &

# 2. Attendre qu'Ollama réponde
echo "Attente du démarrage d'Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 1
done
echo "Ollama prêt ✅"

# 3. Vérifier et télécharger le modèle Mistral si nécessaire (avec cache)
if ! ollama list | grep -q '^mistral'; then
  echo "Téléchargement du modèle mistral..."
  ollama pull mistral
else
  echo "✅ Modèle Mistral déjà disponible"
fi

# 4. Pré-charger les modèles TTS si pas déjà fait
if [ ! -d "/root/.cache/tts" ] || [ -z "$(ls -A /root/.cache/tts 2>/dev/null)" ]; then
  echo "Pré-téléchargement des modèles TTS..."
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
else
  echo "✅ Modèles TTS déjà en cache"
fi

# 5. Démarrer FastAPI
echo "Démarrage du service LLM..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 