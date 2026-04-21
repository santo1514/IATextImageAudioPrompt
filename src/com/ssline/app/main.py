from fastapi import FastAPI, File, UploadFile, Form, HTTPException  # FastAPI: framework web; tipos para recibir archivos y formularios
from typing import Annotated                     # Permite agregar metadatos a los tipos de los parametros
from .processor import process_data              # Importa la funcion principal que ejecuta los modelos de IA

app = FastAPI()                                  # Crea la instancia de la aplicacion FastAPI

@app.post("/process")                            # Define el endpoint POST en la ruta /process
async def executeProcess(
    text: Annotated[str, Form()],                # Texto escrito por el usuario, recibido como campo de formulario
    image: UploadFile = File(...),               # Imagen del producto subida por el usuario
    audio: UploadFile = File(...),               # Audio WAV con la descripcion hablada del usuario
):
    if not image.content_type.startswith("image/"):          # Verifica que el archivo subido sea una imagen valida
        raise HTTPException(status_code=400, detail="Invalid image file type.")  # Rechaza si no es imagen

    if not audio.content_type.startswith("audio/"):          # Verifica que el archivo subido sea audio valido
        raise HTTPException(status_code=400, detail="Invalid audio file type.")  # Rechaza si no es audio

    img_bytes = await image.read()               # Lee los bytes completos de la imagen para pasarlos al procesador
    audio_bytes = await audio.read()             # Lee los bytes completos del audio para pasarlos al procesador

    try:
        resultado = process_data(text, img_bytes, audio_bytes)  # Ejecuta los tres analisis: transcripcion, sentimiento y validacion visual

        return {                                  # Estructura la respuesta JSON con los resultados
            "status": "success",                 # Indica que el procesamiento fue exitoso
            "resultado_ia": {
                "audio_transcrito": resultado["transcripcion"],           # Texto extraido del audio por Whisper
                "sentimiento_detectado": resultado["sentimiento"]         # Etiqueta POS/NEG/NEU y score de confianza
            },
            "validacion_coherencia": {
                "score_coincidencia": resultado["coincidencia_visual_score"],  # Similitud imagen-texto segun CLIP (0-35 aprox)
                "hay_discrepancia": resultado["alerta_discrepancia"],     # True si la imagen no coincide con el audio transcrito
                "mensaje": resultado["mensaje_validacion"]                # Descripcion legible del resultado de validacion
            },
            "dictamenes": resultado["dictamenes"]  # Los tres dictamenes de Claude con su tiempo de respuesta cada uno
        }
    except Exception as e:                       # Captura cualquier error durante el procesamiento
        return {"status": "error", "detail": str(e)}  # Retorna el error como respuesta sin romper la app
