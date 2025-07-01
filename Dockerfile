# --- Étape 1 : dépendances --------------------------------
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime AS base
WORKDIR /app

# Installation des dépendances système...
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python en plusieurs étapes
COPY requirements.txt .

# Installation des dépendances de base
RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    ninja

# Installation des dépendances du projet
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement du modèle TTS (cache dans ~/.local/share/tts)
RUN python - <<'PY'
from TTS.utils.manage import ModelManager
mm = ModelManager()
mm.download_model("tts_models/fr/mai/vits")
mm.download_model("tts_models/fr/css10/vits")
PY

# Copie des fichiers du projet
COPY . .

# --- Installation d'Ollama --------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pré-téléchargement du modèle Mistral 7B (optionnel, ~7 Go)
RUN ollama pull mistral || true      # télécharge si possible, sinon continue

# --- Image finale -----------------------------------------
# On utilise directement la couche 'base'. Les modèles TTS/Whisper
# seront téléchargés au premier démarrage et mis en cache.
EXPOSE 8000
EXPOSE 11434
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"] 