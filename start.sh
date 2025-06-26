#!/bin/bash

# Attendre que Ollama soit prêt
echo "Attente du démarrage d'Ollama..."
until curl -s http://ollama:11434/api/tags > /dev/null; do
    sleep 1
done

# Démarrer le service FastAPI
echo "Démarrage du service LLM..."
exec uvicorn main:app --host 0.0.0.0 --port 3000 --reload 