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
        print(f"[STT] Cargando modelo Whisper: {WHISPER_MODEL}")
        modelo = whisper.load_model(WHISPER_MODEL)
        print(f"[STT] Modelo cargado correctamente")
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

    - Acepta archivos de audio (wav, mp3, m4a, ogg, opus, flac, etc.)
    - La extensión del archivo debe ser correcta (el gateway la detecta)
    - Detecta automáticamente el idioma
    - Retorna texto, idioma detectado y nivel de confianza
    """
    tmp_path = None
    try:
        # Obtener extensión del archivo (el gateway ya detectó el formato)
        extension = os.path.splitext(audio.filename or "audio.ogg")[1] or ".ogg"

        # Leer contenido
        contenido = await audio.read()
        if not contenido:
            raise HTTPException(status_code=400, detail="Archivo de audio vacío")

        print(f"[STT] Recibido: {audio.filename} ({len(contenido)} bytes, ext={extension})")

        # Guardar archivo temporal con la extensión correcta
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(contenido)
            tmp_path = tmp.name

        # Cargar modelo y transcribir
        modelo = cargar_modelo()
        resultado = modelo.transcribe(tmp_path)

        texto = resultado["text"].strip()
        idioma = resultado["language"]
        confianza = resultado.get("language_probability", 0.0)

        print(f"[STT] Transcripción exitosa: idioma={idioma}, texto={texto[:50]}...")

        return TranscripcionResponse(
            texto=texto,
            idioma=idioma,
            confianza=confianza
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[STT] Error al transcribir: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al transcribir: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/detect-language")
async def detectar_idioma(audio: UploadFile = File(...)):
    """
    Detecta el idioma del audio sin transcribir completamente.
    """
    tmp_path = None
    try:
        extension = os.path.splitext(audio.filename or "audio.ogg")[1] or ".ogg"
        contenido = await audio.read()

        if not contenido:
            raise HTTPException(status_code=400, detail="Archivo de audio vacío")

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(contenido)
            tmp_path = tmp.name

        modelo = cargar_modelo()

        # Cargar audio y detectar idioma
        audio_data = whisper.load_audio(tmp_path)
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(modelo.device)
        _, probs = modelo.detect_language(mel)

        idioma = max(probs, key=probs.get)
        return {
            "idioma": idioma,
            "confianza": probs[idioma],
            "probabilidades": dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al detectar idioma: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.on_event("startup")
async def precargar_modelo():
    """Precarga el modelo al iniciar el servicio."""
    cargar_modelo()
