import os
import sys
import shutil
import threading
import time
import requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import librosa

from app.config import settings
from app.security import validar_archivo_audio
from app.utils import calcular_riesgo_y_recomendaciones, enviar_reporte_telegram

VOICE_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "motores", "voice_engine")
if VOICE_ENGINE_DIR not in sys.path:
    sys.path.insert(0, VOICE_ENGINE_DIR)
from app.motores.voice_engine import voice_ai_engine

from app.motores.openai_engine import openai_engine

app = FastAPI(
    title=settings.APP_NAME,
    description="Analizador Forense de Audios contra la Extorsión y el Fraude (Nube)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def iniciar_bot_polling():
    """
    Hebra independiente que realiza Long Polling a la API de Telegram.
    Detecta el comando /start y envía el botón Inline con el Deep Link de retorno 'guardian://wsp?tgid=ID'
    """
    token = getattr(settings, "TELEGRAM_TOKEN", None)
    if not token:
        print("⚠️ [Bot Telegram] Polling omitido: Falta TELEGRAM_TOKEN en configuración.")
        return
    
    print("🤖 [Bot Telegram] Iniciando receptor de comandos /start en segundo plano...")
    offset = 0
    url_get_updates = f"https://api.telegram.org/bot{token}/getUpdates"
    url_send_message = f"https://api.telegram.org/bot{token}/sendMessage"
    
    while True:
        try:
            res = requests.get(f"{url_get_updates}?offset={offset}&timeout=5", timeout=10)
            if res.status_code == 200:
                data = res.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = message.get("chat", {}).get("id")
                    
                    if text and text.startswith("/start") and chat_id:
                        payload = {
                            "chat_id": chat_id,
                            "text": (
                                "🚨 *Módulo de Alertas Seguras GuardIAn*\n\n"
                                "Has activado el chat de seguridad con éxito.\n"
                                "Para completar la vinculación automática en tu aplicación móvil y comenzar "
                                "a recibir las pistas periciales, presiona el siguiente botón de abajo:"
                            ),
                            "parse_mode": "Markdown",
                            "reply_markup": {
                                "inline_keyboard": [[
                                    {
                                        "text": "🚀 Completar Vinculación Automática",
                                        "url": f"guardian://wsp?tgid={chat_id}"
                                    }
                                ]]
                            }
                        }
                        requests.post(url_send_message, json=payload, timeout=5)
                        print(f"✅ [Bot Telegram] Enlace profundo enviado de retorno para chat_id: {chat_id}")
        except Exception as e:
            pass
        time.sleep(1)


@app.on_event("startup")
def al_iniciar_servidor():
    threading.Thread(target=iniciar_bot_polling, daemon=True).start()


@app.get("/")
def verificar_estado():
    return {
        "estado": "Online",
        "proyecto": settings.APP_NAME,
        "motores_cargados": [
            "Motor 1 (voice_engine / XLS-R-SLS Local)",
            "Motor 2 (OpenAI API Transcript)",
            "Motor 3 (GPT-4o-mini Ingeniería Social)",
            "Motor 4 (Riesgo Consolidado)"
        ]
    }


def _analizar_voice_engine_desde_ruta(ruta_audio: str):
    if not voice_ai_engine.listo and voice_ai_engine.error_carga:
        raise HTTPException(status_code=503, detail=voice_ai_engine.error_carga)
    try:
        return voice_ai_engine.analizar_clonacion(ruta_audio)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en voice_engine: {str(exc)}") from exc


@app.post("/api/v1/analisis/forense")
async def analizar_audio_forense(
    file: UploadFile = File(...),
    tipo_origen: str = Form("whatsapp"), 
    telegram_chat_id: Optional[str] = Form(None)
):
    validar_archivo_audio(file)

    nombre_archivo = f"evidencia_{file.filename}"
    ruta_guardado = os.path.join(settings.UPLOAD_DIR, nombre_archivo)

    try:
        with open(ruta_guardado, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al escribir archivo: {str(e)}")

    try:
        # =====================================================
        # MOTOR 1 - Detección de voz sintética (Local)
        # =====================================================
        reporte_forense = _analizar_voice_engine_desde_ruta(ruta_guardado)
        score_voz_ia = reporte_forense.score_riesgo

        # =====================================================
        # MOTOR 2 - Transcripción (OpenAI API)
        # =====================================================
        texto_transcrito = openai_engine.transcribir_audio(ruta_guardado)

        if not texto_transcrito:
            return {
                "error": "El motor de transcripción no detectó voz legible o hubo un fallo de API.",
                "analisis_valido": False
            }

        # =====================================================
        # MOTOR 3 - Ingeniería Social (OpenAI GPT-4o-mini)
        # =====================================================
        analisis_social = openai_engine.analizar_texto_social(texto_transcrito)
        tacticas_detectadas = analisis_social.get("tacticas", [])

        # =====================================================
        # MOTOR 4 - Riesgo Consolidado
        # =====================================================
        analisis_riesgo = calcular_riesgo_y_recomendaciones(score_voz_ia, analisis_social)

        duracion = librosa.get_duration(path=ruta_guardado)
        palabras = len(texto_transcrito.split())
        palabras_seg = round(palabras / duracion, 2) if duracion > 0 else 0.0

        nivel_confianza_dinamico = reporte_forense.nivel_confianza
        if score_voz_ia >= 70 and analisis_social.get("riesgo_social", 0) < 30 and not analisis_social.get("fraude_detectado", False):
            if analisis_social.get("es_narracion_tts", False):
                nivel_confianza_dinamico = "Voz Artificial Detectada (Locución / Lector TTS)"
            else:
                nivel_confianza_dinamico = "Anomalía acústica aislada (Posible compresión Opus / WhatsApp)"

        response_data = {
            "archivo_procesado": file.filename,
            "transcripcion_whisper": texto_transcrito,
            "resumen_seguridad": analisis_riesgo["resumen_seguridad"],
            "metricas": {
                "motor1_voz_ia": score_voz_ia,
                "nivel_confianza_voz": nivel_confianza_dinamico,
                "motor3_ingenieria_social": analisis_social["riesgo_social"],
                "riesgo_global": analisis_riesgo["riesgo_global"],
                "nivel": analisis_riesgo["nivel_evaluacion"]
            },
            "desglose_tacticas": tacticas_detectadas,
            "analisis_social": {
                 "fraude_detectado": analisis_social["fraude_detectado"],
                 "confianza_fraude": analisis_social["confianza_fraude"],
                 "riesgo_social": analisis_social["riesgo_social"],
                 "nivel_riesgo": analisis_social["nivel_riesgo"],
                 "alerta_lexico_local": analisis_social.get("alerta_lexico_local", False),
                 "idioma_detectado": analisis_social.get("idioma_detectado", "Desconocido")
            },
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
            "detalles_audio_whisper": {
                "duracion_physica_archivo_segundos": round(duracion, 2),
                "duracion_actividad_habla_segundos": round(duracion, 2),
                "tiempo_inactividad_silencio_segundos": 0.0,
                "porcentaje_silencio_llamada": 0.0,
                "total_fragmentos_deteccion": 1,
                "densidad_habla_palabras_por_segundo": palabras_seg,
                "segmentacion_lineal": []
            },
            "recomendaciones_seguridad": analisis_riesgo["recomendaciones"]
        }

        riesgo_str = str(analisis_riesgo.get("nivel_evaluacion", "")).upper()
        
        if "CRÍTICO" in riesgo_str or "ALTO" in riesgo_str or "CRITICO" in riesgo_str or analisis_social.get("es_narracion_tts", False) or analisis_social.get("alerta_lexico_local", False):
            enviar_reporte_telegram(
                archivo=file.filename,
                origen=tipo_origen,
                transcripcion=texto_transcrito,
                tacticas=tacticas_detectadas,
                riesgo=riesgo_str,
                recomendaciones=analisis_riesgo["recomendaciones"],
                chat_id_destino=telegram_chat_id,
                score_voz_ia=score_voz_ia,
                nivel_confianza_voz=nivel_confianza_dinamico,
                resumen_seguridad=analisis_riesgo["resumen_seguridad"]
            )

        return response_data

    except Exception as e:
        print(f"❌ [Error Backend] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fallo en motores: {str(e)}")

    finally:
        if os.path.exists(ruta_guardado):
            os.remove(ruta_guardado)