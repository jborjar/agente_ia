# Agente IA

Stack de servicios de inteligencia artificial. Provee APIs stateless para procesamiento de voz, texto e imagenes.

## Servicios

| Servicio | Puerto | Descripcion | Escalable |
|----------|--------|-------------|-----------|
| nginx | 8000/8001/8002/11434 | Load balancer API/STT/TTS/LLM | - |
| api | - | API unificada (texto + audio) | Horizontal |
| stt | - | Speech to Text (Whisper) | Horizontal |
| llm | - | Modelos de lenguaje (Ollama) | Horizontal (max 2) |
| tts | - | Text to Speech (Coqui) | Horizontal |

## Capacidades

### API Unificada

| Caracteristica | Detalle |
|----------------|---------|
| Endpoints | `/chat`, `/voice`, `/image`, `/document`, `/classify` |
| Audio entrada | WAV, OGG, MP3, M4A, FLAC, Opus |
| Imagen entrada | PNG, JPEG, GIF, WebP |
| Documento entrada | PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT |
| Audio salida | OGG Opus base64 (compatible WhatsApp) |
| Respuesta | Siempre `{texto, audio_b64, idioma}` |

**Flujos:**
- `/chat`: texto → LLM → TTS → OGG → respuesta
- `/voice`: audio → STT → LLM → TTS → OGG → respuesta
- `/image`: imagen → LLM Vision → TTS → OGG → respuesta
- `/document`: Office/PDF/imagen → LLM Vision → TTS → OGG → respuesta
- `/classify`: documento → clasificación → TTS → OGG → respuesta

**Nota:** El audio de salida es OGG Opus en base64 porque es el formato nativo de WhatsApp para notas de voz.

### STT (Speech to Text)

| Caracteristica | Detalle |
|----------------|---------|
| Motor | OpenAI Whisper |
| Endpoint | `POST /transcribe` |
| Modelo default | `small` (configurable) |
| Deteccion idioma | Automatica |
| Formatos entrada | WAV, OGG, MP3, M4A, FLAC, Opus, WebM |
| Formato optimo | WAV 16kHz mono PCM |
| RAM aprox | ~2GB (modelo small) |

**Nota:** Whisper acepta cualquier formato de audio gracias a ffmpeg. El formato óptimo es WAV 16kHz mono.

### LLM (Large Language Model)

| Caracteristica | Detalle |
|----------------|---------|
| Motor | Ollama |
| Endpoint | Puerto 11434 (API Ollama estandar) |
| Modelo chat | `qwen2.5:7b` (~5GB RAM) |
| Modelo vision | `llava:7b` (~5GB RAM) |
| Streaming | Soportado |
| Historial | Via parametro `messages` |

### TTS (Text to Speech)

| Caracteristica | Detalle |
|----------------|---------|
| Motor | Coqui TTS |
| Endpoint | `POST /synthesize` |
| Modelo default | `tts_models/es/css10/vits` |
| Idiomas | Espanol (es), Ingles (en), auto-deteccion |
| Formato salida | WAV 22050Hz mono (audio/wav) |
| RAM aprox | ~1-2GB |

**Nota:** TTS genera WAV. La API unificada convierte internamente a OGG Opus para WhatsApp.

### Acceso desde otros stacks

```bash
# Desde cualquier contenedor en la red vpn-proxy
curl http://agente_ia:8000/health   # API unificada
curl http://agente_ia:8001/health   # STT
curl http://agente_ia:8002/health   # TTS
curl http://agente_ia:11434         # LLM
```

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                          AGENTE_IA                               │
│                    (Servicios de IA puros)                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     Nginx (Load Balancer)                    ││
│  │            :8001 (STT)  :8002 (TTS)  :11434 (LLM)           ││
│  └──────────────┬───────────────┬───────────────┬──────────────┘│
│                 │               │               │                │
│      ┌──────────┼─────┐  ┌──────┼─────┐  ┌──────┼─────┐         │
│      v          v     v  v      v     v  v      v     v         │
│  ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐ │
│  │ STT 1 │ │ STT 2 │  │ TTS 1 │ │ TTS 2 │  │ LLM 1 │ │ LLM 2 │ │
│  │Whisper│ │Whisper│  │ Coqui │ │ Coqui │  │Ollama │ │Ollama │ │
│  └───────┘ └───────┘  └───────┘ └───────┘  └───────┘ └───────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │ HTTP
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                         ORQUESTADOR                              │
│                (Webhooks, cola, orquestacion)                    │
│                                                                  │
│    WhatsApp ─► Webhook ─► Redis ─► Workers ─► Respuesta         │
│    Telegram ─►                                                   │
│    Web API ─►                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Instalacion

```bash
# 1. Configurar variables de entorno
cp .env.example .env
nano .env

# 2. Ejecutar script de inicializacion
chmod +x init.sh
./init.sh
```

### Que hace init.sh

| Paso | Descripcion |
|------|-------------|
| 1 | Carga variables de `.env` |
| 2 | Crea directorios de datos |
| 3 | Crea red `vpn-proxy` si no existe |
| 4 | Construye imagenes Docker (stt, tts) |
| 5 | Inicia los 3 servicios (stt, llm, tts) |
| 6 | Descarga modelos de Ollama |
| 7 | Verifica que los servicios respondan |

### Instalacion manual

```bash
docker network create vpn-proxy
docker compose build
docker compose up -d
docker exec llm ollama pull qwen2.5:7b
docker exec llm ollama pull llava:7b
```

## Configuracion

### Variables de Entorno (.env)

```env
# Zona horaria
TZ=America/Mexico_City

# STT - Speech to Text (Whisper)
WHISPER_MODEL=small

# TTS - Text to Speech (Coqui)
COQUI_TTS_MODEL=tts_models/es/css10/vits

# LLM - Modelos Ollama
LLM_CHAT_MODEL=qwen2.5:7b
LLM_IMG_MODEL=llava:7b
LLM_DOCS_MODEL=llava:7b
```

### Modelos Whisper

| Modelo | Tamano | RAM | Velocidad |
|--------|--------|-----|-----------|
| tiny | 39M | ~1GB | Muy rapido |
| base | 74M | ~1GB | Rapido |
| small | 244M | ~2GB | Moderado |
| medium | 769M | ~5GB | Lento |
| large | 1550M | ~10GB | Muy lento |

## APIs de los Servicios

### API Unificada (Recomendada)

La API unificada combina STT, LLM Vision y TTS. Siempre retorna texto + audio OGG Opus (formato nativo de WhatsApp).

**Formatos soportados:**
- **Audio entrada:** WAV, OGG, MP3, M4A, FLAC, Opus, WebM
- **Imagen entrada:** PNG, JPEG, GIF, WebP
- **Documento entrada:** PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, imágenes
- **Audio salida:** OGG Opus en base64 (listo para WhatsApp)

```bash
# Chat (texto → texto + audio OGG)
POST http://agente_ia:8000/chat
Content-Type: application/json
{
  "texto": "Hola, como estas?",
  "idioma": "es",
  "system_prompt": "Eres un asistente amigable"  # opcional
}

# Respuesta
{
  "texto": "Hola! Estoy muy bien, gracias por preguntar.",
  "audio_b64": "T2dnUwACAA...",  # OGG Opus listo para WhatsApp
  "idioma": "es"
}
```

```bash
# Voice (audio cualquier formato → texto + audio OGG)
POST http://agente_ia:8000/voice
Content-Type: multipart/form-data
audio: <archivo de audio (WAV, OGG, MP3, etc.)>

# Respuesta (mismo formato que /chat)
{
  "texto": "Respuesta del asistente...",
  "audio_b64": "T2dnUwACAA...",
  "idioma": "es"
}
```

```bash
# Image (imagen → texto + audio OGG)
POST http://agente_ia:8000/image
Content-Type: multipart/form-data
imagen: <archivo de imagen (PNG, JPEG, etc.)>
prompt: "Describe esta imagen"  # opcional
idioma: "es"  # opcional

# Respuesta (mismo formato)
{
  "texto": "En la imagen se observa...",
  "audio_b64": "T2dnUwACAA...",
  "idioma": "es"
}
```

```bash
# Document (PDF/Office/imagen → texto + audio OGG)
POST http://agente_ia:8000/document
Content-Type: multipart/form-data
archivo: <PDF, DOCX, XLSX, PPTX o imagen>
prompt: "Extrae la información importante"  # opcional
idioma: "es"  # opcional

# Respuesta (analiza todas las páginas)
{
  "texto": "--- Página 1 ---\n...\n--- Página 2 ---\n...",
  "audio_b64": "T2dnUwACAA...",
  "idioma": "es"
}
```

```bash
# Classify (documento → clasificación + audio OGG)
POST http://agente_ia:8000/classify
Content-Type: multipart/form-data
archivo: <PDF, DOCX, XLSX, PPTX o imagen>
idioma: "es"  # opcional

# Respuesta
{
  "tipo_documento": "CSF",
  "confianza": "alta",
  "descripcion": "Constancia de Situación Fiscal del SAT",
  "texto": "Documento identificado como CSF...",
  "audio_b64": "T2dnUwACAA...",
  "idioma": "es"
}
```

**Tipos de documento soportados para clasificación:**
- CSF (Constancia de Situación Fiscal)
- INE / Pasaporte
- Título de propiedad
- Recibo de luz / agua / gas
- Predial
- Acta constitutiva
- Poder notarial
- Comprobante de domicilio
- Estado de cuenta bancario
- CURP / Acta de nacimiento
- Comprobante de ingresos
- Contrato / Factura

```bash
# Health check
GET http://agente_ia:8000/health

# Listar tipos de documento
GET http://agente_ia:8000/tipos-documento
```

### STT - Speech to Text

Acepta cualquier formato de audio (WAV, OGG, MP3, M4A, FLAC, Opus, WebM).

```bash
# Transcribir audio
POST http://stt:8000/transcribe
Content-Type: multipart/form-data
audio: <archivo de audio (cualquier formato)>

# Respuesta
{
  "texto": "Hola como estas",
  "idioma": "es",
  "confianza": 0.95
}
```

```bash
# Health check
GET http://stt:8000/health

# Modelos disponibles
GET http://stt:8000/modelos
```

### LLM - Ollama

```bash
# Chat
POST http://llm:11434/api/chat
Content-Type: application/json
{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "user", "content": "Hola"}
  ],
  "stream": false
}

# Chat con imagen (vision)
POST http://llm:11434/api/chat
{
  "model": "llava:7b",
  "messages": [
    {"role": "user", "content": "Describe la imagen", "images": ["base64..."]}
  ]
}
```

### TTS - Text to Speech

Genera audio en formato WAV 22050Hz mono. Para OGG Opus (WhatsApp), usar la API unificada.

```bash
# Sintetizar texto
POST http://tts:8000/synthesize
Content-Type: application/json
{
  "texto": "Hola, como estas?",
  "idioma": "es"  # opcional, se detecta automaticamente
}

# Respuesta: audio/wav 22050Hz mono (streaming)
```

```bash
# Health check
GET http://tts:8000/health

# Idiomas disponibles
GET http://tts:8000/idiomas
```

## Estructura de Archivos

```
agente_ia/
├── docker-compose.yaml
├── dockerfiles/
│   ├── Dockerfile.api
│   ├── Dockerfile.stt
│   ├── Dockerfile.tts
│   ├── nginx/
│   │   └── nginx.conf
│   └── app/
│       ├── api/requirements.txt
│       ├── stt/requirements.txt
│       └── tts/requirements.txt
├── .env
├── .env.example
├── .gitignore
├── init.sh
├── README.md
└── stack_data/
    ├── api/app/main.py
    ├── stt/app/main.py
    ├── tts/app/main.py
    └── llm/models/
```

## Integracion con Orquestador

Este stack provee unicamente los servicios de IA. Para webhooks, orquestacion y canales de comunicacion (WhatsApp, Telegram, etc.), usar el stack **orquestador**.

```
Usuario -> Orquestador -> agente_ia (STT/LLM/TTS) -> Orquestador -> Usuario
```

Ver documentacion en [orquestador/README.md](../orquestador/README.md)

## Logs y Debug

```bash
# Ver logs de todos los servicios
docker compose logs -f

# Ver logs de un servicio especifico
docker logs -f stt
docker logs -f llm
docker logs -f tts

# Probar health checks
curl http://localhost:8000/health  # desde stt
curl http://localhost:11434        # desde llm
curl http://localhost:8000/health  # desde tts
```

## Troubleshooting

### STT no responde

```bash
# Verificar que el modelo Whisper se cargo
docker logs stt | grep "Cargando modelo"
```

El modelo Whisper tarda en cargar la primera vez (~30 segundos para `small`).

### TTS no responde

```bash
# Verificar que el modelo Coqui se cargo
docker logs tts | grep "modelo"
```

El modelo TTS tambien tarda en cargar la primera vez.

### Modelos de Ollama no disponibles

```bash
# Ver modelos instalados
docker exec llm ollama list

# Descargar modelo
docker exec llm ollama pull qwen2.5:7b
```

### Servicios no se conectan

Verificar que esten en la misma red:

```bash
docker network inspect vpn-proxy
```

## Escalado de Servicios

El stack soporta escalado horizontal para STT, TTS y LLM via nginx load balancer.

### Escalado horizontal

```bash
# Escalar STT a 3 instancias
docker compose up -d --scale stt=3

# Escalar TTS a 2 instancias
docker compose up -d --scale tts=2

# Escalar LLM a 2 instancias (MAXIMO RECOMENDADO)
docker compose up -d --scale llm=2

# Escalar todos
docker compose up -d --scale stt=3 --scale tts=2 --scale llm=2

# Ver instancias activas
docker ps | grep -E "stt|tts|llm"
```

### Consideraciones para LLM

LLM (Ollama) puede escalar horizontalmente pero con restricciones:

| Instancias | RAM Aprox | Recomendado |
|------------|-----------|-------------|
| 1 | ~5-10GB | Uso normal |
| 2 | ~10-20GB | Alta demanda |
| 3+ | ~15-30GB+ | No recomendado |

**Importante:**
- Cada instancia carga los modelos en memoria
- 2 instancias es el maximo recomendado
- Para mas capacidad, usar GPU o modelos mas pequenos

### Puertos del Load Balancer

| Puerto | Servicio | Descripcion |
|--------|----------|-------------|
| 8000 | API | Unificada (texto + audio) - Recomendado |
| 8001 | STT | Balanceado entre instancias |
| 8002 | TTS | Balanceado entre instancias |
| 11434 | LLM | Balanceado entre instancias (max 2) |

### Configuracion del Orquestador

Para usar la API unificada (recomendado):

```env
API_URL=http://agente_ia:8000
```

Para acceso directo a servicios individuales:

```env
STT_URL=http://agente_ia:8001
TTS_URL=http://agente_ia:8002
LLM_URL=http://agente_ia:11434
```
