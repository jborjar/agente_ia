"""
Servicio API Unificado - Orquesta STT, LLM, TTS y RAG (Qdrant)

Endpoints:
- POST /chat: texto → texto + audio (OGG base64)
- POST /voice: audio → texto + audio (OGG base64)
- POST /image: imagen → texto + audio (OGG base64)
- POST /document: PDF/imagen → texto + audio (OGG base64)
- POST /classify: documento → clasificación + texto + audio
- POST /ingest: documento → indexado en base vectorial (RAG)
- POST /ask: pregunta + coleccion → respuesta con contexto recuperado
- GET  /colecciones: lista de colecciones existentes
- DELETE /colecciones/{nombre}: borra una coleccion

Siempre retorna texto y audio en la respuesta.
"""
import os
import io
import re
import uuid
import base64
import hashlib
import tempfile
import subprocess
import httpx
import redis
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
from pdf2image import convert_from_bytes
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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
LLM_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "nomic-embed-text")

# System prompt por defecto
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Eres un asistente útil. Responde de forma concisa.")

# Redis para guardar idioma del usuario
REDIS_URL = os.getenv("REDIS_URL", "redis://:orquestador123@redis-orquestador:6379/0")
HISTORY_TTL = 3600  # 1 hora

# Qdrant (base vectorial para RAG)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "qdrant123")
EMBED_DIM = 768  # dimension de nomic-embed-text
CHUNK_SIZE = 800  # caracteres por chunk
CHUNK_OVERLAP = 100  # solape entre chunks

# Conexión a Redis (lazy init)
_redis_client = None
# Cliente Qdrant (lazy init)
_qdrant_client = None


def get_qdrant() -> QdrantClient:
    """Obtiene cliente Qdrant (lazy init)."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


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


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """Convierte PDF a lista de imágenes (una por página)."""
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
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


async def call_llm(texto: str, system_prompt: str | None = None, temperature: float | None = None) -> str:
    """Llama al servicio LLM para generar respuesta.

    Si se proporciona temperature, se envia en options para controlar
    la aleatoriedad (0.0 = deterministico, 0.8 = default creativo).
    """
    prompt = system_prompt or SYSTEM_PROMPT

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": texto}
        ],
        "stream": False
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

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


# =============================================================
#   RAG - Retrieval Augmented Generation (Qdrant + embeddings)
# =============================================================

class IngestResponse(BaseModel):
    coleccion: str
    doc_id: str
    doc_nombre: str
    doc_hash: str  # SHA256 del archivo (primeros 16 chars visibles)
    chunks_indexados: int
    metodo_extraccion: Optional[str] = None  # "tesseract" o "llava" o "mixto"
    reemplazo: bool  # True si se reemplazo un documento existente con el mismo nombre
    ya_existia: bool  # True si el contenido (hash) ya estaba indexado, no se hizo nada


class AskRequest(BaseModel):
    pregunta: str
    coleccion: str
    top_k: int = 5
    idioma: str = "es"
    # Si se proporciona, limita la busqueda a ese documento dentro de la coleccion
    doc_id: Optional[str] = None
    # Alternativa: limitar por nombre de archivo
    doc_nombre: Optional[str] = None
    # Si es True, la respuesta incluye los chunks que se usaron como contexto
    incluir_fuentes: bool = False


class Fuente(BaseModel):
    doc_id: str
    doc_nombre: str
    chunk_idx: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    texto: str
    idioma: str
    fuentes: Optional[List[Fuente]] = None


class ColeccionInfo(BaseModel):
    nombre: str
    puntos: int


def ocr_image(image_bytes: bytes, lang: str = "spa+eng") -> str:
    """Ejecuta Tesseract OCR sobre una imagen."""
    img = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img, lang=lang)


async def extract_text_smart(file_bytes: bytes, file_type: str) -> tuple[str, str]:
    """
    Extrae texto de un archivo usando Tesseract primero y llava como fallback.
    Retorna (texto, metodo) donde metodo es 'tesseract', 'llava' o 'mixto'.

    Para OCR usa 300 DPI (recomendado por Tesseract para texto pequeno).
    """
    if file_type in ("docx", "doc", "xlsx", "xls", "pptx", "ppt"):
        file_bytes = office_to_pdf(file_bytes, file_type)
        file_type = "pdf"

    paginas_tesseract = 0
    paginas_llava = 0
    textos = []

    if file_type == "pdf":
        # 300 DPI para OCR: mejor precision en texto pequeno y graficos
        images = pdf_to_images(file_bytes, dpi=300)
        if not images:
            raise HTTPException(status_code=400, detail="No se pudieron extraer paginas del PDF")

        for i, img_bytes in enumerate(images):
            texto_ocr = ocr_image(img_bytes).strip()
            # Heuristica: si el OCR salio muy corto o pura basura, usar llava
            if len(texto_ocr) < 50 or _texto_ilegible(texto_ocr):
                image_b64 = image_to_base64(img_bytes)
                prompt = "Extrae TODO el texto visible en este documento, tal cual aparece, sin agregar comentarios."
                texto_vision = await call_llm_vision(image_b64, prompt)
                textos.append(f"[Pagina {i+1}]\n{texto_vision}")
                paginas_llava += 1
            else:
                textos.append(f"[Pagina {i+1}]\n{texto_ocr}")
                paginas_tesseract += 1

    elif file_type in ("png", "jpeg", "gif", "webp"):
        texto_ocr = ocr_image(file_bytes).strip()
        if len(texto_ocr) < 50 or _texto_ilegible(texto_ocr):
            image_b64 = image_to_base64(file_bytes)
            prompt = "Extrae TODO el texto visible en este documento, tal cual aparece, sin agregar comentarios."
            texto_vision = await call_llm_vision(image_b64, prompt)
            textos.append(texto_vision)
            paginas_llava += 1
        else:
            textos.append(texto_ocr)
            paginas_tesseract += 1

    else:
        raise HTTPException(
            status_code=400,
            detail="Tipo de archivo no soportado para ingest. Use PDF, Office o imagenes."
        )

    if paginas_llava > 0 and paginas_tesseract > 0:
        metodo = "mixto"
    elif paginas_llava > 0:
        metodo = "llava"
    else:
        metodo = "tesseract"

    return "\n\n".join(textos), metodo


def _texto_ilegible(texto: str) -> bool:
    """Heuristica: detecta si el OCR salio ilegible (mucho ruido)."""
    if not texto:
        return True
    # Proporcion de caracteres alfanumericos vs total
    alfanum = sum(1 for c in texto if c.isalnum() or c.isspace())
    if len(texto) == 0:
        return True
    return (alfanum / len(texto)) < 0.6


def chunk_text(texto: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Divide texto en chunks con solape, respetando saltos de linea cuando es posible."""
    texto = re.sub(r'\n{3,}', '\n\n', texto).strip()
    if len(texto) <= size:
        return [texto] if texto else []

    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + size
        if fin >= len(texto):
            chunks.append(texto[inicio:].strip())
            break
        # Intentar cortar en salto de linea o punto cercano
        corte = texto.rfind('\n', inicio, fin)
        if corte == -1 or corte < inicio + size // 2:
            corte = texto.rfind('. ', inicio, fin)
        if corte == -1 or corte < inicio + size // 2:
            corte = fin
        else:
            corte += 1
        chunk = texto[inicio:corte].strip()
        if chunk:
            chunks.append(chunk)
        inicio = max(corte - overlap, inicio + 1)
    return chunks


async def get_embedding(texto: str) -> List[float]:
    """Obtiene embedding de un texto via Ollama."""
    payload = {"model": LLM_EMBED_MODEL, "prompt": texto}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{LLM_URL}/api/embeddings", json=payload)
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error en embeddings: {response.text}"
            )
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise HTTPException(status_code=500, detail="Ollama no devolvio embedding")
        return embedding


def _crear_indices_payload(client: QdrantClient, nombre: str):
    """
    Crea indices de payload sobre los campos por los que filtramos en /ask.
    Sin indice, Qdrant puede aplicar filtros inconsistentemente cuando el campo
    tiene baja cardinalidad y devolver vacio.
    """
    for campo in ("doc_id", "doc_nombre", "doc_hash"):
        try:
            client.create_payload_index(
                collection_name=nombre,
                field_name=campo,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Ya existe el indice
            pass


def ensure_collection(nombre: str):
    """
    Crea la coleccion si no existe, con indices de payload sobre doc_id/doc_nombre/doc_hash.
    Si Qdrant tiene basura en disco de un borrado anterior incompleto,
    intenta limpiar y reintenta.
    """
    client = get_qdrant()
    existentes = [c.name for c in client.get_collections().collections]
    if nombre in existentes:
        # Aseguramos que los indices existan (idempotente)
        _crear_indices_payload(client, nombre)
        return

    vectors_config = qmodels.VectorParams(
        size=EMBED_DIM, distance=qmodels.Distance.COSINE
    )
    try:
        client.create_collection(collection_name=nombre, vectors_config=vectors_config)
    except Exception as e:
        # Caso conocido: Qdrant dice "Collection data already exists" porque quedo
        # un directorio huerfano en disco de un borrado anterior. Forzar limpieza.
        if "already exists" in str(e).lower():
            try:
                client.delete_collection(nombre)
            except Exception:
                pass
            client.create_collection(collection_name=nombre, vectors_config=vectors_config)
        else:
            raise

    _crear_indices_payload(client, nombre)


def coleccion_existe(nombre: str) -> bool:
    """Indica si una coleccion existe en Qdrant."""
    client = get_qdrant()
    return nombre in [c.name for c in client.get_collections().collections]


def borrar_puntos_por_payload(coleccion: str, clave: str, valor: str) -> int:
    """
    Borra puntos de una coleccion donde payload[clave] == valor.
    Retorna cuantos puntos habia con esa coincidencia antes de borrar.
    """
    client = get_qdrant()
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key=clave, match=qmodels.MatchValue(value=valor))]
    )
    total = client.count(collection_name=coleccion, count_filter=flt, exact=True).count
    if total > 0:
        client.delete(
            collection_name=coleccion,
            points_selector=qmodels.FilterSelector(filter=flt),
        )
    return total


def contar_puntos_por_payload(coleccion: str, clave: str, valor: str) -> int:
    """Cuenta puntos en una coleccion donde payload[clave] == valor."""
    client = get_qdrant()
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key=clave, match=qmodels.MatchValue(value=valor))]
    )
    return client.count(collection_name=coleccion, count_filter=flt, exact=True).count


def buscar_documento_por_hash(coleccion: str, doc_hash: str) -> Optional[dict]:
    """
    Busca un documento por hash de contenido en una coleccion.
    Retorna un dict con doc_id, doc_nombre y chunks (cantidad) o None si no existe.
    """
    if not coleccion_existe(coleccion):
        return None
    client = get_qdrant()
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_hash", match=qmodels.MatchValue(value=doc_hash))]
    )
    puntos, _ = client.scroll(
        collection_name=coleccion,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
    )
    if not puntos:
        return None
    payload = puntos[0].payload or {}
    doc_id = payload.get("doc_id")
    if not doc_id:
        return None
    chunks = contar_puntos_por_payload(coleccion, "doc_id", doc_id)
    return {
        "doc_id": doc_id,
        "doc_nombre": payload.get("doc_nombre"),
        "chunks": chunks,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    archivo: UploadFile = File(...),
    coleccion: str = Form(...),
):
    """
    Indexa un documento en la base vectorial.

    Flujo: archivo → extraccion texto (Tesseract + fallback llava) → chunking
           → embeddings (Ollama) → Qdrant.
    """
    try:
        file_bytes = await archivo.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Archivo vacio")

        nombre_archivo = archivo.filename or "documento"
        file_type = detect_file_type(file_bytes)
        doc_hash = hashlib.sha256(file_bytes).hexdigest()

        # 1. Asegurar coleccion (necesario antes de poder buscar por hash)
        ensure_collection(coleccion)

        # 2. Si el contenido (hash) ya existe, NO reprocesar: devolver el doc_id existente
        existente = buscar_documento_por_hash(coleccion, doc_hash)
        if existente:
            return IngestResponse(
                coleccion=coleccion,
                doc_id=existente["doc_id"],
                doc_nombre=existente["doc_nombre"] or nombre_archivo,
                doc_hash=doc_hash[:16],
                chunks_indexados=existente["chunks"],
                metodo_extraccion=None,
                reemplazo=False,
                ya_existia=True,
            )

        # 3. Extraer texto
        texto, metodo = await extract_text_smart(file_bytes, file_type)
        if not texto.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del documento")

        # 4. Chunking
        chunks = chunk_text(texto)
        if not chunks:
            raise HTTPException(status_code=400, detail="No se generaron chunks del texto")

        # 5. Si el nombre coincide con un documento existente (pero el contenido cambio),
        #    reemplazar al viejo: es una version actualizada del archivo.
        client = get_qdrant()
        reemplazo = False
        if contar_puntos_por_payload(coleccion, "doc_nombre", nombre_archivo) > 0:
            flt = qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_nombre", match=qmodels.MatchValue(value=nombre_archivo))]
            )
            client.delete(
                collection_name=coleccion,
                points_selector=qmodels.FilterSelector(filter=flt),
            )
            reemplazo = True

        # 6. Embeddings + inserción
        doc_id = str(uuid.uuid4())
        puntos = []
        for idx, chunk in enumerate(chunks):
            vector = await get_embedding(chunk)
            puntos.append(qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "doc_nombre": nombre_archivo,
                    "doc_hash": doc_hash,
                    "chunk_idx": idx,
                    "texto": chunk,
                }
            ))
        client.upsert(collection_name=coleccion, points=puntos)

        return IngestResponse(
            coleccion=coleccion,
            doc_id=doc_id,
            doc_nombre=nombre_archivo,
            doc_hash=doc_hash[:16],
            chunks_indexados=len(puntos),
            metodo_extraccion=metodo,
            reemplazo=reemplazo,
            ya_existia=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ingest: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Responde una pregunta usando RAG sobre una coleccion.

    Flujo: pregunta → embedding → busqueda Qdrant → prompt con contexto → LLM → TTS.
    """
    if not request.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia")

    try:
        client = get_qdrant()
        existentes = [c.name for c in client.get_collections().collections]
        if request.coleccion not in existentes:
            raise HTTPException(
                status_code=404,
                detail=f"La coleccion '{request.coleccion}' no existe"
            )

        # 1. Limpieza de filtros (ignoramos placeholder "string" de Swagger)
        doc_id_clean = (request.doc_id or "").strip()
        doc_nombre_clean = (request.doc_nombre or "").strip()
        if doc_id_clean.lower() == "string":
            doc_id_clean = ""
        if doc_nombre_clean.lower() == "string":
            doc_nombre_clean = ""

        filtros = []
        if doc_id_clean:
            filtros.append(qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=doc_id_clean)
            ))
        if doc_nombre_clean:
            filtros.append(qmodels.FieldCondition(
                key="doc_nombre", match=qmodels.MatchValue(value=doc_nombre_clean)
            ))
        query_filter = qmodels.Filter(must=filtros) if filtros else None

        # 2. Decidir estrategia de recuperacion:
        #    - Si se filtra por documento Y el doc tiene <= MAX_CHUNKS_FULL: traer
        #      TODOS los chunks (mejor recall, evitamos que el embedding ranquee mal).
        #    - Si no, busqueda semantica con top_k.
        MAX_CHUNKS_FULL = 30
        usar_todo_el_doc = False
        if query_filter:
            total_chunks_doc = client.count(
                collection_name=request.coleccion,
                count_filter=query_filter,
                exact=True,
            ).count
            if 0 < total_chunks_doc <= MAX_CHUNKS_FULL:
                usar_todo_el_doc = True

        if usar_todo_el_doc:
            # Traer TODOS los chunks del documento ordenados por chunk_idx
            puntos, _ = client.scroll(
                collection_name=request.coleccion,
                scroll_filter=query_filter,
                limit=MAX_CHUNKS_FULL,
                with_payload=True,
            )
            # Wrapper para que el codigo siguiente trate igual ambos casos
            class _R:
                __slots__ = ("payload", "score")
                def __init__(self, payload):
                    self.payload = payload
                    self.score = 1.0
            resultados = [_R(p.payload or {}) for p in puntos]
            # Ordenar por chunk_idx para mantener el orden original del documento
            resultados.sort(key=lambda r: r.payload.get("chunk_idx", 0))
        else:
            # Busqueda semantica clasica
            vector = await get_embedding(request.pregunta)
            respuesta_busqueda = client.query_points(
                collection_name=request.coleccion,
                query=vector,
                limit=request.top_k,
                with_payload=True,
                query_filter=query_filter,
            )
            resultados = respuesta_busqueda.points

        if not resultados:
            raise HTTPException(
                status_code=404,
                detail="No se encontro contexto relevante en la coleccion"
            )

        # 3. Armar contexto y fuentes
        bloques = []
        fuentes = []
        for r in resultados:
            p = r.payload or {}
            texto_chunk = p.get("texto", "")
            bloques.append(texto_chunk)
            snippet = texto_chunk[:200] + ("..." if len(texto_chunk) > 200 else "")
            fuentes.append(Fuente(
                doc_id=p.get("doc_id", ""),
                doc_nombre=p.get("doc_nombre", ""),
                chunk_idx=p.get("chunk_idx", 0),
                score=float(r.score),
                snippet=snippet,
            ))

        contexto = "\n---\n".join(bloques)
        system_prompt = (
            "Extrae del contexto el VALOR del campo pedido. NO incluyas la etiqueta del campo.\n"
            "Ejemplo: si dice 'Numero Exterior: MANZANA 171 LOTE 19', "
            "respondes: MANZANA 171 LOTE 19 (NO 'Numero Exterior: MANZANA 171 LOTE 19').\n"
            "Equivalencias:\n"
            "- razon social = 'Nombre, denominacion o razon social'\n"
            "- calle = 'Nombre de Vialidad'\n"
            "- colonia = 'Nombre de la Colonia'\n"
            "- municipio = 'Nombre del Municipio o Demarcacion Territorial'\n"
            "- estado = 'Nombre de la Entidad Federativa'\n"
            "- RFC = 'Registro Federal de Contribuyentes'\n"
            "- CURP = 'Clave Unica de Registro de Poblacion'\n"
            "IGNORA 'Entre Calle' y 'Y Calle' (son colindancias, no la calle principal).\n"
            "Si el dato esta en tabla, extrae el renglon de datos.\n"
            "Respeta mayusculas/minusculas y acentos del documento.\n"
            "FORMATO DE RESPUESTA OBLIGATORIO:\n"
            "- Si encuentras el valor: devuelve UNICAMENTE el valor, sin parentesis, "
            "sin comentarios, sin 'No se encuentra'.\n"
            "- Si NO lo encuentras: devuelve UNICAMENTE 'No se encuentra'.\n"
            "NUNCA combines el valor con 'No se encuentra' en la misma respuesta."
        )
        user_prompt = f"{contexto}\n\nPregunta: {request.pregunta}"

        # temperature=0.1 = casi deterministico, pero sin volverse demasiado conservador
        # (con 0.0 el modelo a veces responde 'No se encuentra' cuando el dato si existe)
        respuesta_texto = await call_llm(user_prompt, system_prompt, temperature=0.1)
        if not respuesta_texto:
            raise HTTPException(status_code=500, detail="LLM no genero respuesta")

        # Limpieza defensiva: el LLM a veces ignora la regla y mezcla el valor con
        # comentarios entre parentesis o sufijos como '(No se encuentra)'. Los quitamos.
        respuesta_texto = re.sub(r'\s*\([^)]*[Nn]o\s+se\s+encuentra[^)]*\)\s*$', '', respuesta_texto)
        respuesta_texto = re.sub(r'\s*\(No se encuentra\)\s*', '', respuesta_texto)
        respuesta_texto = respuesta_texto.strip()

        return AskResponse(
            texto=respuesta_texto,
            idioma=request.idioma,
            fuentes=fuentes if request.incluir_fuentes else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ask: {str(e)}")


@app.get("/colecciones", response_model=List[ColeccionInfo])
async def listar_colecciones():
    """Lista las colecciones disponibles en Qdrant."""
    try:
        client = get_qdrant()
        colecciones = client.get_collections().collections
        resultado = []
        for c in colecciones:
            info = client.get_collection(c.name)
            resultado.append(ColeccionInfo(nombre=c.name, puntos=info.points_count or 0))
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando colecciones: {str(e)}")


@app.delete("/colecciones/{nombre}")
async def borrar_coleccion(nombre: str):
    """Borra una coleccion completa."""
    try:
        client = get_qdrant()
        existentes = [c.name for c in client.get_collections().collections]
        if nombre not in existentes:
            raise HTTPException(status_code=404, detail=f"La coleccion '{nombre}' no existe")
        client.delete_collection(nombre)
        return {"status": "ok", "coleccion": nombre, "accion": "eliminada"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error borrando coleccion: {str(e)}")


@app.get("/colecciones/{nombre}/documentos")
async def listar_documentos(nombre: str):
    """Lista los documentos indexados en una coleccion (agrupados por doc_id)."""
    try:
        if not coleccion_existe(nombre):
            raise HTTPException(status_code=404, detail=f"La coleccion '{nombre}' no existe")
        client = get_qdrant()
        documentos: dict[str, dict] = {}
        # Recorrer todos los puntos en lotes
        next_offset = None
        while True:
            puntos, next_offset = client.scroll(
                collection_name=nombre,
                limit=256,
                with_payload=True,
                offset=next_offset,
            )
            for p in puntos:
                payload = p.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                if doc_id not in documentos:
                    documentos[doc_id] = {
                        "doc_id": doc_id,
                        "doc_nombre": payload.get("doc_nombre"),
                        "doc_hash": (payload.get("doc_hash") or "")[:16],
                        "chunks": 0,
                    }
                documentos[doc_id]["chunks"] += 1
            if next_offset is None:
                break
        return {"coleccion": nombre, "documentos": list(documentos.values())}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando documentos: {str(e)}")


@app.delete("/colecciones/{nombre}/documentos/{doc_id}")
async def borrar_documento(nombre: str, doc_id: str):
    """Borra un documento individual de la coleccion (todos sus chunks)."""
    try:
        if not coleccion_existe(nombre):
            raise HTTPException(status_code=404, detail=f"La coleccion '{nombre}' no existe")
        borrados = borrar_puntos_por_payload(nombre, "doc_id", doc_id)
        if borrados == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontro el documento '{doc_id}' en la coleccion '{nombre}'"
            )
        return {
            "status": "ok",
            "coleccion": nombre,
            "doc_id": doc_id,
            "chunks_eliminados": borrados,
            "accion": "eliminado",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error borrando documento: {str(e)}")
