FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des fichiers du projet
COPY . .

# Téléchargement des modèles
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/fr/css10/vits')"
RUN python -c "import whisper; whisper.load_model('base')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"] 