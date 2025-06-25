from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from TTS.api import TTS
import os
import uuid
import torch

class TTSRequest(BaseModel):
    text: str
    language: str = "fr"

class TTSService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTS(model_name="tts_models/fr/css10/vits").to(self.device)
        os.makedirs("/app/audio", exist_ok=True)

    async def generate_speech(self, text: str, language: str = "fr") -> str:
        try:
            # Générer un nom de fichier unique
            filename = f"{uuid.uuid4()}.wav"
            output_path = f"/app/audio/{filename}"

            # Générer l'audio
            self.model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=None,
                language=language
            )

            return filename
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Créer l'application FastAPI
app = FastAPI()

# Créer une instance du service TTS
tts_service = TTSService()

# Monter le répertoire audio pour servir les fichiers statiques
app.mount("/audio", StaticFiles(directory="/app/audio"), name="audio")

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        filename = await tts_service.generate_speech(request.text, request.language)
        return {"audio_url": f"/audio/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 