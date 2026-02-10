"""
Servicio STT - Speech to Text con Whisper
"""
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import whisper

app = FastAPI(
    title="STT Service",
    description="Servicio de transcripción de voz a texto con Whisper"
)

# Configuración del modelo
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Modelo cargado
modelo = None


class TranscripcionResponse(BaseModel):
    texto: str
    idioma: str
    confianza: float


def cargar_modelo():
    """Carga el modelo Whisper."""
    global modelo
    if modelo is None:
        modelo = whisper.load_model(WHISPER_MODEL)
    return modelo


@app.get("/health")
async def health():
    """Endpoint de salud del servicio."""
    return {"status": "ok", "servicio": "stt", "modelo": WHISPER_MODEL}


@app.get("/modelos")
async def modelos_disponibles():
    """Lista los modelos disponibles de Whisper."""
    return {
        "modelo_actual": WHISPER_MODEL,
        "modelos": ["tiny", "base", "small", "medium", "large"]
    }


@app.post("/transcribe", response_model=TranscripcionResponse)
async def transcribir(audio: UploadFile = File(...)):
    """
    Transcribe audio a texto.

    - Acepta archivos de audio (wav, mp3, m4a, etc.)
    - Detecta automáticamente el idioma
    - Retorna texto, idioma detectado y nivel de confianza
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó archivo de audio")

    try:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contenido = await audio.read()
            tmp.write(contenido)
            tmp_path = tmp.name

        # Cargar modelo y transcribir
        modelo = cargar_modelo()
        resultado = modelo.transcribe(tmp_path)

        # Limpiar archivo temporal
        os.unlink(tmp_path)

        return TranscripcionResponse(
            texto=resultado["text"].strip(),
            idioma=resultado["language"],
            confianza=resultado.get("language_probability", 0.0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al transcribir: {str(e)}")


@app.post("/detect-language")
async def detectar_idioma(audio: UploadFile = File(...)):
    """
    Detecta el idioma del audio sin transcribir completamente.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó archivo de audio")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contenido = await audio.read()
            tmp.write(contenido)
            tmp_path = tmp.name

        modelo = cargar_modelo()

        # Cargar audio y detectar idioma
        audio_data = whisper.load_audio(tmp_path)
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(modelo.device)
        _, probs = modelo.detect_language(mel)

        os.unlink(tmp_path)

        idioma = max(probs, key=probs.get)
        return {
            "idioma": idioma,
            "confianza": probs[idioma],
            "probabilidades": dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al detectar idioma: {str(e)}")


@app.on_event("startup")
async def precargar_modelo():
    """Precarga el modelo al iniciar el servicio."""
    cargar_modelo()
