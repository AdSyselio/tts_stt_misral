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
# ➜ S'assurer que le modèle est dispo
if ! ollama list | grep -q '^mistral'; then
  echo "Téléchargement du modèle mistral..."
  ollama pull mistral
fi
# 3. Démarrer FastAPI
echo "Démarrage du service LLM..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 