import whisper                                   # Libreria de OpenAI para transcribir audio a texto
import torch                                     # Framework de deep learning, necesario para detectar GPU
import io                                        # Permite tratar bytes en memoria como si fueran un archivo
import os                                        # Lee la variable de entorno COHERE_API_KEY
import time                                      # Mide el tiempo de cada llamada a Cohere
import cohere                                    # SDK oficial de Cohere para llamar a la API de Command R
from PIL import Image                            # Abre y convierte imagenes desde bytes
from transformers import pipeline, CLIPProcessor, CLIPModel  # Modelos de HuggingFace para sentimiento y vision

print("Cargando modelos de IA...")               # Mensaje de inicio, visible en consola al arrancar la app

device = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU si esta disponible, sino CPU

model_whisper = whisper.load_model("base", device=device)  # Carga Whisper "base": liviano y suficiente para español

sentiment_analyzer = pipeline(                   # Crea el pipeline de analisis de sentimiento
    "sentiment-analysis",                        # Tarea: clasificar sentimiento del texto
    model="pysentimiento/robertuito-sentiment-analysis",  # Modelo entrenado en español nativo, devuelve POS/NEG/NEU
    device=device                                # Ejecuta en GPU o CPU segun disponibilidad
)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # Modelo CLIP: compara imagen y texto en un espacio vectorial compartido
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")     # Preprocesador de CLIP: tokeniza texto y normaliza imagen

cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])  # Cliente V2 de Cohere: usa la variable de entorno COHERE_API_KEY

# --- TECNICA 1: ZERO-SHOT ---
# Sin ejemplos: Cohere infiere el formato solo a partir de la instruccion
PROMPT_ZERO_SHOT = """
Eres un auditor de reclamos de productos dañados.

Datos del analisis automatico:
- Sentimiento del reclamo: {sentimiento_label} (confianza: {sentimiento_score:.2f})
- Transcripcion del audio: {transcripcion}
- Imagen coincide con el reclamo: {coincide}
- Score de coincidencia visual: {score}

Emite un dictamen breve: indica SI o NO se debe aprobar el reembolso y justifica en una sola oracion.
"""

# --- TECNICA 2: FEW-SHOT ---
# Con 3 ejemplos reales: Claude aprende el patron de formato y criterio esperado
PROMPT_FEW_SHOT = """
Eres un auditor de reclamos de productos dañados.

Ejemplos de dictamenes anteriores:

Caso 1:
- Sentimiento: NEG (0.91) | Transcripcion: "el telefono llego con la pantalla rota"
- Imagen coincide: SI | Score: 24.14
- Dictamen: SI se aprueba el reembolso. La imagen confirma el daño descrito y el reclamo es genuinamente negativo.

Caso 2:
- Sentimiento: NEG (0.78) | Transcripcion: "el cargador no funciona"
- Imagen coincide: NO | Score: 20.98
- Dictamen: NO se aprueba el reembolso. La imagen no corresponde al daño descrito, lo que sugiere inconsistencia en el reclamo.

Caso 3:
- Sentimiento: POS (0.85) | Transcripcion: "el producto llego perfecto"
- Imagen coincide: SI | Score: 23.50
- Dictamen: NO se aprueba el reembolso. El reclamo indica satisfaccion con el producto, no hay daño que compensar.

Ahora analiza este caso:
- Sentimiento: {sentimiento_label} ({sentimiento_score:.2f}) | Transcripcion: "{transcripcion}"
- Imagen coincide: {coincide} | Score: {score}

Dictamen:"""

# --- TECNICA 3: CHAIN-OF-THOUGHT ---
# Razonamiento paso a paso: reduce errores en casos ambiguos o con señales contradictorias
PROMPT_COT = """
Eres un auditor de reclamos de productos dañados.

Datos del analisis automatico:
- Sentimiento del reclamo: {sentimiento_label} (confianza: {sentimiento_score:.2f})
- Transcripcion del audio: {transcripcion}
- Imagen coincide con el reclamo: {coincide}
- Score de coincidencia visual: {score}

Razona paso a paso antes de concluir:
1. ¿El sentimiento indica un reclamo genuino?
2. ¿La imagen es coherente con lo que describe el audio?
3. ¿Hay señales de inconsistencia o posible fraude?
4. Conclusion final: SI o NO aprobar el reembolso en una sola oracion.
"""

def _llamar_cohere(prompt: str) -> tuple[str, float]:  # Llama a Cohere y retorna (texto, segundos_transcurridos)
    inicio = time.time()                         # Marca el tiempo antes de enviar la solicitud
    response = cohere_client.chat(
        model="command-a-03-2025",                  # Usando 'command-a-03-2025' como un modelo actualizado y disponible
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512                           # CoT puede necesitar mas tokens que zero-shot
    )
    duracion = round(time.time() - inicio, 2)    # Calcula cuantos segundos tomo la respuesta
    return response.message.content[0].text, duracion  # Retorna el texto y el tiempo medido

def process_data(text_input: str, image_bytes: bytes, audio_bytes: bytes):  # Funcion principal que ejecuta los cuatro analisis

    # --- A. TRANSCRIPCION DE AUDIO ---
    with open("temp_audio.wav", "wb") as f:      # Whisper no acepta bytes en memoria, requiere archivo en disco
        f.write(audio_bytes)                     # Escribe el audio recibido como archivo temporal WAV

    audio_result = model_whisper.transcribe("temp_audio.wav")  # Transcribe el audio a texto
    transcribed_text = audio_result['text'].strip()            # Extrae solo el texto y elimina espacios sobrantes

    # --- B. ANALISIS DE SENTIMIENTO ---
    combined_text = f"{text_input}. {transcribed_text}"  # Une el texto escrito y el audio transcrito en uno solo
    sentiment = sentiment_analyzer(combined_text)[0]     # Analiza el sentimiento, devuelve {"label": "NEG/POS/NEU", "score": 0.0-1.0}

    # --- C. VALIDACION IMAGEN VS TEXTO (CLIP) ---
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Decodifica los bytes de la imagen y la convierte a RGB

    inputs = clip_processor(                     # Prepara imagen y texto para ser procesados por CLIP
        text=[transcribed_text],                 # Solo se usa el audio transcrito para comparar contra la imagen
        images=image,                            # Imagen del producto recibida
        return_tensors="pt",                     # Devuelve tensores de PyTorch compatibles con el modelo
        padding=True                             # Rellena el texto al largo maximo del batch
    ).to(device)                                 # Mueve los tensores al mismo dispositivo que el modelo

    outputs = clip_model(**inputs)               # Ejecuta CLIP: calcula similitud entre imagen y texto
    relevance_score = outputs.logits_per_image[0][0].item()  # Score raw sin softmax: con un solo texto softmax devolveria siempre 1.0

    # Umbral calibrado con casos reales: imagen rota + texto "rota" = 24.14, imagen buena + texto "rota" = 20.98
    COINCIDENCIA_UMBRAL = 22.5                   # 22.5 separa ambos casos con margen suficiente
    alerta = relevance_score < COINCIDENCIA_UMBRAL   # True si la imagen no coincide con el texto del audio
    mensaje_alerta = (
        "ALERTA: La imagen no parece representar lo que se describe en el texto o el audio."
        if alerta else
        "La imagen coincide con el contexto."
    )

    # Argumentos comunes que se inyectan en los tres prompts
    datos = dict(
        sentimiento_label=sentiment["label"],
        sentimiento_score=sentiment["score"],
        transcripcion=transcribed_text,
        coincide="NO" if alerta else "SI",       # Convierte el booleano a texto legible para Claude
        score=round(relevance_score, 4)
    )

    # --- D. DICTAMEN FINAL CON TRES TECNICAS DE PROMPTING ---
    dictamen_zero_shot, tiempo_zero_shot = _llamar_cohere(PROMPT_ZERO_SHOT.format(**datos))   # Tecnica 1
    dictamen_few_shot, tiempo_few_shot   = _llamar_cohere(PROMPT_FEW_SHOT.format(**datos))    # Tecnica 2
    dictamen_cot,      tiempo_cot        = _llamar_cohere(PROMPT_COT.format(**datos))         # Tecnica 3

    return {                                     # Retorna todos los resultados incluyendo los tres dictamenes
        "transcripcion": transcribed_text,       # Texto extraido del audio
        "sentimiento": sentiment,                # Resultado del analisis de sentimiento
        "coincidencia_visual_score": round(relevance_score, 4),  # Score de similitud imagen-texto
        "alerta_discrepancia": alerta,           # True si hay discrepancia entre imagen y texto
        "mensaje_validacion": mensaje_alerta,    # Descripcion legible del resultado de validacion
        "dictamenes": {                          # Agrupa los tres dictamenes para comparacion
            "zero_shot": {
                "texto": dictamen_zero_shot,     # Respuesta de Cohere sin ejemplos ni razonamiento guiado
                "tiempo_segundos": tiempo_zero_shot  # Tiempo que tomo esta llamada
            },
            "few_shot": {
                "texto": dictamen_few_shot,      # Respuesta de Cohere guiada por tres ejemplos concretos
                "tiempo_segundos": tiempo_few_shot
            },
            "chain_of_thought": {
                "texto": dictamen_cot,           # Respuesta de Cohere con razonamiento paso a paso explicito
                "tiempo_segundos": tiempo_cot
            }
        }
    }
