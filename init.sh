#!/bin/bash
# =============================================================
# Script de inicializacion para Agente IA
# =============================================================
# Configura automaticamente:
# - Directorios de datos
# - Construccion de imagenes Docker
# - Descarga de modelos de Ollama
# - Verificacion de servicios
# =============================================================

set -e

echo "=== Inicializando stack agente_ia ==="

# Cargar variables de entorno
if [ -f .env ]; then
    source .env
else
    echo "Creando .env desde .env.example..."
    cp .env.example .env
    source .env
fi

# Crear directorios de datos
echo "Creando directorios de datos..."
mkdir -p stack_data/{stt/app,llm/models,tts/app}

# Verificar red externa
if ! docker network inspect vpn-proxy >/dev/null 2>&1; then
    echo "Creando red vpn-proxy..."
    docker network create vpn-proxy
fi

# Construir imagenes
echo "Construyendo imagenes..."
docker compose build

# Iniciar servicios
echo "Iniciando servicios..."
docker compose up -d

# =============================================================
# OLLAMA - Descargar modelos
# =============================================================
echo ""
echo "=== Descargando modelos de Ollama ==="
echo "Esperando a que Ollama este listo..."
until docker exec llm ollama list >/dev/null 2>&1; do
    sleep 2
done
echo "Ollama listo."

# Modelo de chat
LLM_CHAT=${LLM_CHAT_MODEL:-qwen2.5:7b}
echo "Descargando modelo de chat: $LLM_CHAT"
docker exec llm ollama pull "$LLM_CHAT" || echo "Modelo $LLM_CHAT ya existe o error"

# Modelo de vision (si es diferente)
LLM_IMG=${LLM_IMG_MODEL:-llava:7b}
if [ "$LLM_IMG" != "$LLM_CHAT" ]; then
    echo "Descargando modelo de vision: $LLM_IMG"
    docker exec llm ollama pull "$LLM_IMG" || echo "Modelo $LLM_IMG ya existe o error"
fi

# Modelo de documentos (si es diferente)
LLM_DOCS=${LLM_DOCS_MODEL:-llava:7b}
if [ "$LLM_DOCS" != "$LLM_CHAT" ] && [ "$LLM_DOCS" != "$LLM_IMG" ]; then
    echo "Descargando modelo de documentos: $LLM_DOCS"
    docker exec llm ollama pull "$LLM_DOCS" || echo "Modelo $LLM_DOCS ya existe o error"
fi

echo "Modelos instalados:"
docker exec llm ollama list

# =============================================================
# VERIFICAR SERVICIOS
# =============================================================
echo ""
echo "=== Verificando servicios ==="

# Esperar a que STT este listo
echo "Esperando a que STT este listo..."
STT_TRIES=0
until docker exec stt curl -s http://localhost:8000/health >/dev/null 2>&1; do
    sleep 3
    STT_TRIES=$((STT_TRIES + 1))
    if [ $STT_TRIES -gt 20 ]; then
        echo "STT tarda en iniciar (cargando modelo Whisper), continuando..."
        break
    fi
done

# Esperar a que TTS este listo
echo "Esperando a que TTS este listo..."
TTS_TRIES=0
until docker exec tts curl -s http://localhost:8000/health >/dev/null 2>&1; do
    sleep 3
    TTS_TRIES=$((TTS_TRIES + 1))
    if [ $TTS_TRIES -gt 20 ]; then
        echo "TTS tarda en iniciar (cargando modelo Coqui), continuando..."
        break
    fi
done

# =============================================================
# RESUMEN FINAL
# =============================================================
echo ""
echo "=========================================="
echo "   AGENTE IA - CONFIGURACION COMPLETA"
echo "=========================================="
echo ""
docker compose ps
echo ""
echo "Servicios de IA:"
echo "  - stt (Whisper): POST http://stt:8000/transcribe"
echo "  - llm (Ollama):  POST http://llm:11434/api/chat"
echo "  - tts (Coqui):   POST http://tts:8000/synthesize"
echo ""
echo "Modelos configurados:"
echo "  - Chat: $LLM_CHAT"
echo "  - Vision: $LLM_IMG"
echo "  - Documentos: $LLM_DOCS"
echo ""
echo "NOTA: Este stack solo provee servicios de IA."
echo "      Para webhooks y orquestacion, usar el stack orchestrator."
echo ""
echo "=========================================="
