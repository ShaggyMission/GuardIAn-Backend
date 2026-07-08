import os
import sys
import shutil
import json  
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends  
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session  

from app.config import settings
from app.security import validar_archivo_audio
from app.utils import calcular_riesgo_y_recomendaciones

# 💾 Importaciones de tu Capa de Persistencia Forense
from app.database import engine, Base, get_db
from app.models import Evidencia

# El motor nuevo vive en app/motores/voice_engine/ y usa imports internos planos.
VOICE_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "motores", "voice_engine")
if VOICE_ENGINE_DIR not in sys.path:
    sys.path.insert(0, VOICE_ENGINE_DIR)

# Importación de las instancias reales de los motores de IA
from app.motores.whisper_engine import whisper_engine
from app.motores.social_engine import social_engine

# Motor 1 centralizado en el paquete nuevo app/motores/voice_engine/
from app.motores.voice_engine import voice_ai_engine

# ⚙️ Inicialización Automática de Infraestructura Local
# Al encender el servidor, verifica o crea de forma transparente el archivo callshield.db
Base.metadata.create_all(bind=engine)

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


@app.get("/api/v1/voice-engine/health")
def health_voice_engine():
    return voice_ai_engine.estado()


def _analizar_voice_engine_desde_ruta(ruta_audio: str):
    if not voice_ai_engine.listo and voice_ai_engine.error_carga:
        raise HTTPException(status_code=503, detail=voice_ai_engine.error_carga)

    try:
        reporte = voice_ai_engine.analizar_clonacion(ruta_audio)
        return reporte
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error en voice_engine: {str(exc)}"
        ) from exc


@app.post("/api/v1/analisis/forense")
async def analizar_audio_forense(
    file: UploadFile = File(...),
    uuid_dispositivo: str = Form(...),  
    db: Session = Depends(get_db)       
):
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
        reporte_forense = _analizar_voice_engine_desde_ruta(ruta_guardado)

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
        analisis_social = social_engine.analizar_texto(texto_transcrito)
        tacticas_detectadas = analisis_social["tacticas"]

        # =====================================================
        # MOTOR 4 - Riesgo Consolidado
        # =====================================================
        analisis_riesgo = calcular_riesgo_y_recomendaciones(score_voz_ia, analisis_social)

        # 🌟 Estructura Consolidada Optimizada (Mantiene al 100% las métricas de tus compañeros)
        response_data = {
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
            # Se respetan todas las variables añadidas por tu equipo para la defensa
            "analisis_forense": {
                "modelo": reporte_forense.evidencia_neuronal.nombre_modelo,
                "modelo_disponible": reporte_forense.evidencia_neuronal.disponible,
                "score_modelo": reporte_forense.evidencia_neuronal.score_fake_pct,
                "prediccion": reporte_forense.prediccion,
                "prob_real": reporte_forense.prob_real,
                "prob_fake": reporte_forense.prob_fake,
                "margen_decision": reporte_forense.margen_decision,
                "umbral_decision": reporte_forense.umbral_decision,
                "checkpoint": reporte_forense.checkpoint,
                "dispositivo": reporte_forense.dispositivo,
                "configuracion_modelo": reporte_forense.configuracion,
                "advertencia": reporte_forense.advertencia
            },
            "detalles_audio_whisper": whisper_engine.obtener_metricas_forenses(),
            "recomendaciones_seguridad": analisis_riesgo["recomendaciones"]
        }

        # =====================================================
        # 🛡️ PERSISTENCIA FORENSE EN CALIENTE (Estrategia No Invasiva)
        # =====================================================
        try:
            nueva_evidencia = Evidencia(
                uuid_dispositivo=uuid_dispositivo,
                nombre_archivo=file.filename,
                resultado_json=json.dumps(response_data, ensure_ascii=False)
            )
            db.add(nueva_evidencia)
            db.commit()
            db.refresh(nueva_evidencia)
        except Exception as db_err:
            print(f"⚠️ [Advertencia de Resguardo Forense] Falló el guardado local: {str(db_err)}")

        return response_data

    except Exception as e:
        print(f"❌ [Fallo General Backend] Error en la tubería forense: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Fallo en los motores analíticos: {str(e)}"
        )

    finally:
        # Limpieza preventiva del archivo
        if os.path.exists(ruta_guardado):
            os.remove(ruta_guardado)


# =====================================================
# 📱 ENDPOINT: HISTORIAL LOCAL POR DISPOSITIVO
# =====================================================
@app.get("/api/v1/analisis/historial/{uuid_dispositivo}")
def obtener_historial_dispositivo(uuid_dispositivo: str, db: Session = Depends(get_db)):
    """
    Recupera y reconstruye cronológicamente todas las evidencias resguardadas
    pertenecientes al hardware del dispositivo móvil consultante.
    """
    registros = (
        db.query(Evidencia)
        .filter(Evidencia.uuid_dispositivo == uuid_dispositivo)
        .order_by(Evidencia.fecha_registro.desc())
        .all()
    )
    
    historial = []
    for reg in registros:
        historial.append({
            "id": reg.id,
            "uuid_dispositivo": reg.uuid_dispositivo,
            "nombre_archivo": reg.nombre_archivo,
            "fecha_registro": reg.fecha_registro,
            "resultado": json.loads(reg.resultado_json)
        })
    return historial


# =====================================================
# 🔍 ENDPOINT: PISTA DE AUDITORÍA FORENSE GLOBAL
# =====================================================
@app.get("/api/v1/analisis/auditoria")
def obtener_auditoria_global(db: Session = Depends(get_db)):
    """
    Endpoint maestro sin filtros de procedencia. Permite a los peritos del sistema
    auditar, extraer y reconstruir la cadena de custodia completa del backend.
    """
    registros = db.query(Evidencia).order_by(Evidencia.fecha_registro.desc()).all()
    
    auditoria = []
    for reg in registros:
        auditoria.append({
            "id": reg.id,
            "uuid_dispositivo": reg.uuid_dispositivo,
            "nombre_archivo": reg.nombre_archivo,
            "fecha_registro": reg.fecha_registro,
            "resultado": json.loads(reg.resultado_json)
        })
    return auditoria