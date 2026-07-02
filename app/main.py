import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.security import validar_archivo_audio
from app.utils import calcular_riesgo_y_recomendaciones

# Importación de las instancias reales de los motores de IA
from app.motores.whisper_engine import whisper_engine
from app.motores.social_engine import social_engine
from app.motores.voice_ai_engine import voice_ai_engine

app = FastAPI(
    title=settings.APP_NAME,
    description="Analizador Forense de Audios contra la Extorsión y el Fraude"
)

# Configuración de CORS para permitir conexiones desde la app móvil (Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def verificar_estado():
    return {
        "estado": "Online",
        "proyecto": settings.APP_NAME,
        "motores_cargados": [
            "Motor 1 (Wav2Vec2-Bert-VoiceDetector)",
            "Motor 2 (Whisper-Tiny)",
            "Motor 3 (BART-ZeroShot)"
        ]
    }


@app.post("/api/v1/analisis/forense")
async def analizar_audio_forense(file: UploadFile = File(...)):
    # 🛡️ Validar extensión y tipo de archivo
    validar_archivo_audio(file)

    # Definir rutas para el procesamiento seguro del archivo
    nombre_archivo = f"evidencia_{file.filename}"
    ruta_guardado = os.path.join(settings.UPLOAD_DIR, nombre_archivo)

    # Asegurar que el directorio de cargas exista
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Guardar el archivo
    try:
        with open(ruta_guardado, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al escribir el archivo en el servidor: {str(e)}"
        )

    try:
        # =====================================================
        # MOTOR 1 - Detección de voz sintética
        # =====================================================
        reporte_forense = voice_ai_engine.analizar_clonacion(ruta_guardado)

        # Extraemos el score para mantener compatibilidad
        score_voz_ia = reporte_forense.score_riesgo

        # =====================================================
        # MOTOR 2 - Whisper
        # =====================================================
        texto_transcrito = whisper_engine.transcribir(ruta_guardado)

        if not texto_transcrito:
            return {
                "error": "El motor de transcripción no detectó voz legible en el archivo.",
                "analisis_valido": False
            }

        # =====================================================
        # MOTOR 3 - Ingeniería Social
        # =====================================================
        tacticas_detectadas = social_engine.analizar_texto(texto_transcrito)

        # =====================================================
        # MOTOR 4 - Riesgo Consolidado
        # =====================================================
        analisis_riesgo = calcular_riesgo_y_recomendaciones(
            score_voz_ia,
            tacticas_detectadas
        )

        return {
            "archivo_procesado": file.filename,

            "transcripcion_whisper": texto_transcrito,

            "metricas": {
                "motor1_voz_ia": score_voz_ia,
                "nivel_confianza_voz": reporte_forense.nivel_confianza,
                "motor3_ingenieria_social": max(tacticas_detectadas.values()) if tacticas_detectadas else 0,
                "riesgo_global": analisis_riesgo["riesgo_global"],
                "nivel": analisis_riesgo["nivel_evaluacion"]
            },

            "desglose_tacticas": tacticas_detectadas,

            # Información adicional del análisis forense
            "analisis_forense": {
                "modelo": reporte_forense.evidencia_neuronal.nombre_modelo,
                "modelo_disponible": reporte_forense.evidencia_neuronal.disponible,
                "score_modelo": reporte_forense.evidencia_neuronal.score_fake_pct,

                "benford": {
                    "mad": reporte_forense.evidencia_benford.mad,
                    "p_value": reporte_forense.evidencia_benford.p_value,
                    "categoria": reporte_forense.evidencia_benford.categoria_conformidad
                },

                "entropia": {
                    "valor_bits": reporte_forense.evidencia_entropia.valor_bits
                },

                "advertencia": reporte_forense.advertencia
            },

            "recomendaciones_seguridad": analisis_riesgo["recomendaciones"]
        }

    except Exception as e:
        print(f"❌ [Fallo General Backend] Error en la tubería forense: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Fallo en los motores analíticos: {str(e)}"
        )

    finally:
        # Limpieza preventiva
        if os.path.exists(ruta_guardado):
            os.remove(ruta_guardado)