"""
Servicio TTS - Text to Speech con detección automática de idioma
"""
import os
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langdetect import detect, LangDetectException
from TTS.api import TTS

app = FastAPI(
    title="TTS Service",
    description="Servicio de síntesis de voz con detección automática de idioma"
)

# Modelos por idioma
MODELOS_TTS = {
    "es": "tts_models/es/css10/vits",
    "en": "tts_models/en/ljspeech/vits",
    "de": "tts_models/de/thorsten/vits",
    "fr": "tts_models/fr/css10/vits",
    "pt": "tts_models/pt/cv/vits",
    "it": "tts_models/it/mai_female/vits",
}

# Modelo por defecto desde variable de entorno
MODELO_DEFAULT = os.getenv("COQUI_TTS_MODEL", "tts_models/es/css10/vits")

# Cache de modelos cargados
modelos_cargados = {}


class TextoEntrada(BaseModel):
    texto: str
    idioma: str | None = None  # Si no se especifica, se detecta automáticamente


def obtener_modelo(idioma: str) -> tuple[TTS, str]:
    """Obtiene o carga el modelo TTS para el idioma especificado."""
    # Si el idioma no está soportado, usar español
    if idioma not in MODELOS_TTS:
        idioma = "es"

    modelo_nombre = MODELOS_TTS[idioma]

    if modelo_nombre not in modelos_cargados:
        modelos_cargados[modelo_nombre] = TTS(modelo_nombre)

    return modelos_cargados[modelo_nombre], idioma


def detectar_idioma(texto: str) -> str:
    """Detecta el idioma del texto. Si falla, retorna español."""
    try:
        idioma = detect(texto)
        # Si no está soportado, usar español
        return idioma if idioma in MODELOS_TTS else "es"
    except LangDetectException:
        return "es"


@app.get("/health")
async def health():
    """Endpoint de salud del servicio."""
    return {"status": "ok", "servicio": "tts"}


@app.get("/idiomas")
async def idiomas_disponibles():
    """Lista los idiomas disponibles."""
    return {
        "idiomas": list(MODELOS_TTS.keys()),
        "modelo_default": MODELO_DEFAULT
    }


@app.post("/synthesize")
async def sintetizar(entrada: TextoEntrada):
    """
    Sintetiza texto a voz.

    - Si no se especifica idioma, se detecta automáticamente
    - Retorna audio en formato WAV
    """
    if not entrada.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    # Detectar o usar idioma especificado
    idioma_solicitado = entrada.idioma or detectar_idioma(entrada.texto)

    try:
        tts, idioma = obtener_modelo(idioma_solicitado)

        # Generar audio en memoria
        buffer = io.BytesIO()
        tts.tts_to_file(text=entrada.texto, file_path=buffer)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Detected-Language": idioma,
                "Content-Disposition": "attachment; filename=audio.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al sintetizar: {str(e)}")


@app.on_event("startup")
async def cargar_modelo_default():
    """Precarga el modelo de español al iniciar."""
    obtener_modelo("es")
