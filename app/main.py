import os
import sys
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.security import validar_archivo_audio
from app.utils import calcular_riesgo_y_recomendaciones

# El motor nuevo vive en app/motores/voice_engine/ y usa imports internos planos.
VOICE_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "motores", "voice_engine")
if VOICE_ENGINE_DIR not in sys.path:
    sys.path.insert(0, VOICE_ENGINE_DIR)

# Importación de las instancias reales de los motores de IA
from app.motores.whisper_engine import whisper_engine
from app.motores.social_engine import social_engine

# Motor 1 centralizado en el paquete nuevo app/motores/voice_engine/
from app.motores.voice_engine import voice_ai_engine


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
            "Motor 1 (voice_engine / XLS-R-SLS)",
            "Motor 2 (Whisper-Tiny)",
            "Motor 3 (mDeBERTa Ingeniería Social)",
            "Motor 4 (Riesgo Consolidado)"
        ]
    }


def _guardar_audio_temporal_voice_engine(file: UploadFile) -> str:
    nombre_archivo = Path(file.filename or "audio").name
    temp_dir = Path(tempfile.gettempdir()) / "callshield_voice_engine"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ruta_guardado = temp_dir / f"voice_engine_{nombre_archivo}"

    with ruta_guardado.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(ruta_guardado)


@app.get("/api/v1/voice-engine/health")
def health_voice_engine():
    return voice_ai_engine.estado()


@app.post("/api/v1/voice-engine/analizar")
async def analizar_audio_voice_engine(file: UploadFile = File(...)):
    validar_archivo_audio(file)

    ruta_guardada = _guardar_audio_temporal_voice_engine(file)

    try:
        if not voice_ai_engine.listo and voice_ai_engine.error_carga:
            raise HTTPException(status_code=503, detail=voice_ai_engine.error_carga)

        reporte = voice_ai_engine.analizar_clonacion(ruta_guardada)
        return reporte.to_dict()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error en voice_engine: {str(exc)}"
        ) from exc
    finally:
        if os.path.exists(ruta_guardada):
            os.remove(ruta_guardada)


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
        analisis_social = social_engine.analizar_texto(
            texto_transcrito
        )


        tacticas_detectadas = analisis_social["tacticas"]

        # =====================================================
        # MOTOR 4 - Riesgo Consolidado
        # =====================================================
        analisis_riesgo = calcular_riesgo_y_recomendaciones(
            score_voz_ia,
            analisis_social
        )

        return {
            "archivo_procesado": file.filename,

            "transcripcion_whisper": texto_transcrito,

            "metricas": {
                "motor1_voz_ia": score_voz_ia,
                "nivel_confianza_voz": reporte_forense.nivel_confianza,
                "motor3_ingenieria_social": analisis_social["riesgo_social"],
                "riesgo_global": analisis_riesgo["riesgo_global"],
                "nivel": analisis_riesgo["nivel_evaluacion"]
            },

            "desglose_tacticas": tacticas_detectadas,
            "analisis_social": {

                 "fraude_detectado": analisis_social["fraude_detectado"],

                 "confianza_fraude": analisis_social["confianza_fraude"],

                 "riesgo_social": analisis_social["riesgo_social"],

                 "nivel_riesgo": analisis_social["nivel_riesgo"]

        },

            # Información adicional del análisis forense
            "analisis_forense": {
                "modelo": reporte_forense.evidencia_neuronal.nombre_modelo,
                "modelo_disponible": reporte_forense.evidencia_neuronal.disponible,
                "score_modelo": reporte_forense.evidencia_neuronal.score_fake_pct,

                "advertencia": reporte_forense.advertencia
            },
            
            "detalles_audio_whisper": whisper_engine.obtener_metricas_forenses(),
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
