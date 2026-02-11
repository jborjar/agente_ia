"""
Servicio API Unificado - Orquesta STT, LLM y TTS

Endpoints:
- POST /chat: texto → texto + audio (OGG base64)
- POST /voice: audio → texto + audio (OGG base64)
- POST /image: imagen → texto + audio (OGG base64)
- POST /document: PDF/imagen → texto + audio (OGG base64)
- POST /classify: documento → clasificación + texto + audio

Siempre retorna texto y audio en la respuesta.
"""
import os
import io
import base64
import tempfile
import subprocess
import httpx
import redis
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
from pdf2image import convert_from_bytes
from PIL import Image

app = FastAPI(
    title="API Unificada",
    description="Orquesta STT, LLM y TTS. Siempre retorna texto + audio."
)

# URLs de servicios internos (sin balanceador, acceso directo)
STT_URL = os.getenv("STT_URL", "http://stt:8000")
TTS_URL = os.getenv("TTS_URL", "http://tts:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:11434")

# Modelos de LLM
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_IMG_MODEL = os.getenv("LLM_IMG_MODEL", "llava:7b")

# System prompt por defecto
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Eres un asistente útil. Responde de forma concisa.")

# Redis para guardar idioma del usuario
REDIS_URL = os.getenv("REDIS_URL", "redis://:orquestador123@redis-orquestador:6379/0")
HISTORY_TTL = 3600  # 1 hora

# Conexión a Redis (lazy init)
_redis_client = None


def get_redis() -> redis.Redis:
    """Obtiene conexión a Redis (lazy init)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def set_user_language(channel: str, user_id: str, language: str):
    """Guarda el idioma del usuario en Redis."""
    try:
        r = get_redis()
        key = f"chat:language:{channel}:{user_id}"
        r.setex(key, HISTORY_TTL, language)
    except Exception as e:
        print(f"[API] Error guardando idioma en Redis: {e}")


# Timeout para llamadas HTTP (5 minutos para LLM)
TIMEOUT = httpx.Timeout(300.0, connect=30.0)

# Tipos de documento para clasificación
TIPOS_DOCUMENTO = [
    "CSF",                    # Constancia de Situación Fiscal
    "INE",                    # Credencial INE/IFE
    "Pasaporte",
    "Título de propiedad",
    "Recibo de luz",
    "Recibo de agua",
    "Recibo de gas",
    "Predial",
    "Acta constitutiva",
    "Poder notarial",
    "Comprobante de domicilio",
    "Estado de cuenta bancario",
    "CURP",
    "Acta de nacimiento",
    "Comprobante de ingresos",
    "Contrato",
    "Factura",
    "Otro"
]

PROMPT_CLASIFICACION = f"""Analiza este documento y clasifícalo en una de las siguientes categorías:
{', '.join(TIPOS_DOCUMENTO)}

Responde SOLO con el formato:
TIPO: [categoría]
CONFIANZA: [alta/media/baja]
DESCRIPCION: [breve descripción del documento]

Si no puedes identificar el documento, usa "Otro" como tipo."""


class ChatRequest(BaseModel):
    texto: str
    idioma: str | None = None
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    texto: str
    audio_b64: str
    idioma: str


class ImageRequest(BaseModel):
    prompt: str | None = None
    idioma: str | None = None


class ClassifyResponse(BaseModel):
    tipo_documento: str
    confianza: str
    descripcion: str
    texto: str
    audio_b64: str
    idioma: str


def wav_to_ogg_base64(wav_bytes: bytes) -> str:
    """Convierte WAV a OGG Opus y retorna en base64."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_file.write(wav_bytes)
        wav_path = wav_file.name

    ogg_path = wav_path.replace(".wav", ".ogg")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-c:a", "libopus", "-b:a", "48k",
            "-application", "voip",
            ogg_path
        ], capture_output=True, check=True)

        with open(ogg_path, "rb") as f:
            ogg_bytes = f.read()

        return base64.b64encode(ogg_bytes).decode("utf-8")
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if os.path.exists(ogg_path):
            os.unlink(ogg_path)


def image_to_base64(image_bytes: bytes) -> str:
    """Convierte imagen a base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Convierte PDF a lista de imágenes (una por página)."""
    images = convert_from_bytes(pdf_bytes, dpi=150)
    result = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        result.append(buffer.getvalue())
    return result


def detect_file_type(file_bytes: bytes) -> str:
    """Detecta el tipo de archivo por magic bytes."""
    if file_bytes[:4] == b'%PDF':
        return "pdf"
    elif file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "png"
    elif file_bytes[:2] == b'\xff\xd8':
        return "jpeg"
    elif file_bytes[:4] == b'GIF8':
        return "gif"
    elif file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WEBP':
        return "webp"
    # Archivos Office (ZIP con estructura específica)
    elif file_bytes[:4] == b'PK\x03\x04':
        # Es un archivo ZIP, verificar si es Office
        if b'word/' in file_bytes[:2000]:
            return "docx"
        elif b'xl/' in file_bytes[:2000]:
            return "xlsx"
        elif b'ppt/' in file_bytes[:2000]:
            return "pptx"
        else:
            return "zip"
    # Archivos Office antiguos (OLE2)
    elif file_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return "doc"  # Puede ser doc, xls, ppt
    else:
        return "unknown"


def office_to_pdf(file_bytes: bytes, file_type: str) -> bytes:
    """Convierte archivo Office a PDF usando LibreOffice."""
    # Mapear extensiones
    extensions = {
        "docx": ".docx", "doc": ".doc",
        "xlsx": ".xlsx", "xls": ".xls",
        "pptx": ".pptx", "ppt": ".ppt"
    }
    ext = extensions.get(file_type, ".docx")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar archivo original
        input_path = os.path.join(tmpdir, f"input{ext}")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        # Convertir a PDF con LibreOffice
        result = subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", tmpdir, input_path
        ], capture_output=True, timeout=120)

        if result.returncode != 0:
            raise Exception(f"Error convirtiendo Office a PDF: {result.stderr.decode()}")

        # Leer PDF generado
        pdf_path = os.path.join(tmpdir, "input.pdf")
        if not os.path.exists(pdf_path):
            raise Exception("No se generó el PDF")

        with open(pdf_path, "rb") as f:
            return f.read()


async def call_stt(audio_bytes: bytes, filename: str = "audio.ogg") -> dict:
    """Llama al servicio STT para transcribir audio."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        files = {"audio": (filename, audio_bytes)}
        response = await client.post(f"{STT_URL}/transcribe", files=files)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error en STT: {response.text}"
            )

        return response.json()


async def call_llm(texto: str, system_prompt: str | None = None) -> str:
    """Llama al servicio LLM para generar respuesta."""
    prompt = system_prompt or SYSTEM_PROMPT

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": texto}
        ],
        "stream": False
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{LLM_URL}/api/chat", json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error en LLM: {response.text}"
            )

        data = response.json()
        return data.get("message", {}).get("content", "")


async def call_llm_vision(
    image_b64: str,
    prompt: str = "Describe esta imagen en detalle.",
    system_prompt: str | None = None
) -> str:
    """Llama al servicio LLM con modelo de visión para analizar imagen."""
    sys_prompt = system_prompt or "Eres un asistente experto en análisis de imágenes y documentos."

    payload = {
        "model": LLM_IMG_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt, "images": [image_b64]}
        ],
        "stream": False
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{LLM_URL}/api/chat", json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error en LLM Vision: {response.text}"
            )

        data = response.json()
        return data.get("message", {}).get("content", "")


async def call_tts(texto: str, idioma: str = "es") -> bytes:
    """Llama al servicio TTS para sintetizar audio."""
    payload = {"texto": texto, "idioma": idioma}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{TTS_URL}/synthesize", json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error en TTS: {response.text}"
            )

        return response.content


def parse_clasificacion(texto: str) -> tuple[str, str, str]:
    """Parsea la respuesta de clasificación del LLM."""
    tipo = "Otro"
    confianza = "baja"
    descripcion = texto

    lines = texto.strip().split("\n")
    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith("TIPO:"):
            tipo = line.split(":", 1)[1].strip()
        elif line_upper.startswith("CONFIANZA:"):
            confianza = line.split(":", 1)[1].strip().lower()
        elif line_upper.startswith("DESCRIPCION:") or line_upper.startswith("DESCRIPCIÓN:"):
            descripcion = line.split(":", 1)[1].strip()

    # Validar tipo
    if tipo not in TIPOS_DOCUMENTO:
        tipo = "Otro"

    return tipo, confianza, descripcion


@app.get("/health")
async def health():
    """Endpoint de salud del servicio."""
    return {
        "status": "ok",
        "servicio": "api",
        "modelos": {
            "chat": LLM_MODEL,
            "vision": LLM_IMG_MODEL
        },
        "servicios": {
            "stt": STT_URL,
            "tts": TTS_URL,
            "llm": LLM_URL
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Procesa texto y retorna texto + audio.

    Flujo: texto → LLM → TTS → respuesta
    """
    if not request.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    idioma = request.idioma or "es"

    try:
        respuesta_texto = await call_llm(request.texto, request.system_prompt)

        if not respuesta_texto:
            raise HTTPException(status_code=500, detail="LLM no generó respuesta")

        wav_bytes = await call_tts(respuesta_texto, idioma)
        audio_b64 = wav_to_ogg_base64(wav_bytes)

        return ChatResponse(
            texto=respuesta_texto,
            audio_b64=audio_b64,
            idioma=idioma
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando chat: {str(e)}")


@app.post("/voice", response_model=ChatResponse)
async def voice(audio: UploadFile = File(...)):
    """
    Procesa audio y retorna texto + audio.

    Flujo: audio → STT → LLM → TTS → respuesta
    """
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Archivo de audio vacío")

        filename = audio.filename or "audio.ogg"

        stt_result = await call_stt(audio_bytes, filename)
        texto_usuario = stt_result.get("texto", "")
        idioma = stt_result.get("idioma", "es")

        if not texto_usuario:
            raise HTTPException(status_code=400, detail="No se pudo transcribir el audio")

        respuesta_texto = await call_llm(texto_usuario)

        if not respuesta_texto:
            raise HTTPException(status_code=500, detail="LLM no generó respuesta")

        wav_bytes = await call_tts(respuesta_texto, idioma)
        audio_b64 = wav_to_ogg_base64(wav_bytes)

        return ChatResponse(
            texto=respuesta_texto,
            audio_b64=audio_b64,
            idioma=idioma
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando voice: {str(e)}")


@app.post("/image", response_model=ChatResponse)
async def image(
    imagen: UploadFile = File(...),
    prompt: str = Form(default="Describe esta imagen en detalle."),
    idioma: str = Form(default="es")
):
    """
    Procesa imagen y retorna texto + audio.

    Flujo: imagen → LLM Vision → TTS → respuesta
    """
    try:
        image_bytes = await imagen.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Archivo de imagen vacío")

        image_b64 = image_to_base64(image_bytes)

        respuesta_texto = await call_llm_vision(image_b64, prompt)

        if not respuesta_texto:
            raise HTTPException(status_code=500, detail="LLM Vision no generó respuesta")

        wav_bytes = await call_tts(respuesta_texto, idioma)
        audio_b64 = wav_to_ogg_base64(wav_bytes)

        return ChatResponse(
            texto=respuesta_texto,
            audio_b64=audio_b64,
            idioma=idioma
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.post("/document", response_model=ChatResponse)
async def document(
    archivo: UploadFile = File(...),
    prompt: str = Form(default="Analiza este documento y extrae la información importante."),
    idioma: str = Form(default="es")
):
    """
    Procesa documento (PDF, Office o imagen) y retorna texto + audio.

    Formatos soportados:
    - PDF
    - Office: DOCX, DOC, XLSX, XLS, PPTX, PPT
    - Imágenes: PNG, JPEG, GIF, WebP

    Flujo: documento → (Office→PDF→imágenes) → LLM Vision → TTS → respuesta
    """
    try:
        file_bytes = await archivo.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        file_type = detect_file_type(file_bytes)

        # Si es archivo Office, convertir a PDF primero
        if file_type in ("docx", "doc", "xlsx", "xls", "pptx", "ppt"):
            file_bytes = office_to_pdf(file_bytes, file_type)
            file_type = "pdf"

        if file_type == "pdf":
            # Convertir PDF a imágenes y analizar cada página
            images = pdf_to_images(file_bytes)
            if not images:
                raise HTTPException(status_code=400, detail="No se pudieron extraer páginas del PDF")

            resultados = []
            for i, img_bytes in enumerate(images):
                image_b64 = image_to_base64(img_bytes)
                page_prompt = f"Página {i+1}: {prompt}"
                resultado = await call_llm_vision(image_b64, page_prompt)
                resultados.append(f"--- Página {i+1} ---\n{resultado}")

            respuesta_texto = "\n\n".join(resultados)

        elif file_type in ("png", "jpeg", "gif", "webp"):
            # Es una imagen, analizar directamente
            image_b64 = image_to_base64(file_bytes)
            respuesta_texto = await call_llm_vision(image_b64, prompt)

        else:
            raise HTTPException(
                status_code=400,
                detail="Tipo de archivo no soportado. Use PDF, Office (DOCX, XLSX, PPTX) o imágenes"
            )

        if not respuesta_texto:
            raise HTTPException(status_code=500, detail="LLM Vision no generó respuesta")

        wav_bytes = await call_tts(respuesta_texto, idioma)
        audio_b64 = wav_to_ogg_base64(wav_bytes)

        return ChatResponse(
            texto=respuesta_texto,
            audio_b64=audio_b64,
            idioma=idioma
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    archivo: UploadFile = File(...),
    idioma: str = Form(default="es")
):
    """
    Clasifica un documento mexicano.

    Formatos soportados:
    - PDF
    - Office: DOCX, DOC, XLSX, XLS, PPTX, PPT
    - Imágenes: PNG, JPEG, GIF, WebP

    Tipos de documento:
    - CSF (Constancia de Situación Fiscal)
    - INE / Pasaporte
    - Título de propiedad
    - Recibo de luz / agua / gas
    - Predial
    - Acta constitutiva
    - Comprobante de domicilio
    - Estado de cuenta bancario
    - CURP / Acta de nacimiento
    - Y más...

    Flujo: documento → (Office→PDF) → LLM Vision → clasificación → TTS → respuesta
    """
    try:
        file_bytes = await archivo.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        file_type = detect_file_type(file_bytes)

        # Si es archivo Office, convertir a PDF primero
        if file_type in ("docx", "doc", "xlsx", "xls", "pptx", "ppt"):
            file_bytes = office_to_pdf(file_bytes, file_type)
            file_type = "pdf"

        if file_type == "pdf":
            # Para clasificación, solo analizamos la primera página
            images = pdf_to_images(file_bytes)
            if not images:
                raise HTTPException(status_code=400, detail="No se pudieron extraer páginas del PDF")
            image_b64 = image_to_base64(images[0])

        elif file_type in ("png", "jpeg", "gif", "webp"):
            image_b64 = image_to_base64(file_bytes)

        else:
            raise HTTPException(
                status_code=400,
                detail="Tipo de archivo no soportado. Use PDF, Office (DOCX, XLSX, PPTX) o imágenes"
            )

        # Clasificar documento
        respuesta_raw = await call_llm_vision(image_b64, PROMPT_CLASIFICACION)

        if not respuesta_raw:
            raise HTTPException(status_code=500, detail="LLM Vision no generó respuesta")

        tipo, confianza, descripcion = parse_clasificacion(respuesta_raw)

        # Generar texto para audio
        texto_audio = f"Documento identificado como {tipo}. {descripcion}"

        wav_bytes = await call_tts(texto_audio, idioma)
        audio_b64 = wav_to_ogg_base64(wav_bytes)

        return ClassifyResponse(
            tipo_documento=tipo,
            confianza=confianza,
            descripcion=descripcion,
            texto=texto_audio,
            audio_b64=audio_b64,
            idioma=idioma
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clasificando documento: {str(e)}")


@app.get("/tipos-documento")
async def tipos_documento():
    """Lista los tipos de documento que se pueden clasificar."""
    return {"tipos": TIPOS_DOCUMENTO}


# =============================================================
#   ENDPOINT LLM CON DETECCIÓN DE IDIOMA
# =============================================================

class LLMChatRequest(BaseModel):
    """Request para /llm_chat"""
    messages: List[dict]
    model: str = "qwen2.5:7b"
    channel: str = "whatsapp"
    user_id: str


class LLMMessage(BaseModel):
    """Mensaje de respuesta con idioma"""
    role: str
    content: str
    language: str


class LLMChatResponse(BaseModel):
    """Response de /llm_chat - misma estructura que Ollama + language"""
    model: str
    created_at: str
    message: LLMMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@app.post("/llm_chat", response_model=LLMChatResponse)
async def llm_chat(request: LLMChatRequest):
    """Chat con LLM + detección de idioma usando langdetect."""
    from langdetect import detect, LangDetectException

    # Llamar a Ollama /api/chat sin modificar los mensajes
    payload = {
        "model": request.model,
        "messages": request.messages,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{LLM_URL}/api/chat", json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error en LLM: {response.text}")
        data = response.json()

    # Extraer content
    content = data.get("message", {}).get("content", "").strip()

    # Detectar idioma con langdetect
    try:
        language = detect(content) if content else "es"
    except LangDetectException:
        language = "es"

    print(f"[API] LLM response language detected: {language}")

    # Guardar idioma en Redis
    set_user_language(request.channel, request.user_id, language)

    # Retornar misma estructura de Ollama + language
    return LLMChatResponse(
        model=data.get("model", request.model),
        created_at=data.get("created_at", ""),
        message=LLMMessage(
            role=data.get("message", {}).get("role", "assistant"),
            content=content,
            language=language
        ),
        done=data.get("done", True),
        total_duration=data.get("total_duration"),
        load_duration=data.get("load_duration"),
        prompt_eval_count=data.get("prompt_eval_count"),
        prompt_eval_duration=data.get("prompt_eval_duration"),
        eval_count=data.get("eval_count"),
        eval_duration=data.get("eval_duration")
    )
