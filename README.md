# Lab2 - Analisis de Imagen, Texto y Audio

## Prerequisitos

Antes de ejecutar el proyecto asegurate de tener instalado lo siguiente:

- **Python 3.10 o superior** — [Descargar Python](https://www.python.org/downloads/)
- **pip** — Incluido con Python 3.4+
- **Git** — Para clonar el repositorio

## Instalacion del entorno

1. Crear el entorno virtual:
```bash
python -m venv .venv
```

2. Activar el entorno virtual:
```bash
source .venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install fastapi uvicorn python-multipart openai-whisper transformers torch torchvision torchaudio Pillow cohere
```

### Descripcion de las dependencias

| Paquete | Uso |
|---|---|
| `fastapi` | Framework web para construir la API REST |
| `uvicorn` | Servidor ASGI para ejecutar FastAPI |
| `python-multipart` | Necesario para recibir archivos y formularios en FastAPI |
| `openai-whisper` | Modelo de OpenAI para transcribir audio a texto |
| `transformers` | Libreria de HuggingFace con modelos de IA (sentimiento y CLIP) |
| `torch` `torchvision` `torchaudio` | Framework de deep learning requerido por Whisper y Transformers |
| `Pillow` | Procesamiento de imagenes |
