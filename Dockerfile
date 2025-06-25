FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    python3-dev \
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

# Copie des fichiers du projet
COPY . .

# Téléchargement des modèles
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/fr/css10/vits')"
RUN python -c "import whisper; whisper.load_model('base')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"] 