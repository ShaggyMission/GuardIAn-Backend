import os
import requests
from app.config import settings

def enviar_alerta_telegram(archivo: str, riesgo: str, recomendaciones: list, tipo_origen: str, chat_id_destino: str = None):
    """
    Envía una notificación enriquecida al bot de Telegram. Soporta Chat ID dinámico.
    Diferencia el origen del análisis e incluye el desglose de recomendaciones de seguridad.
    """
    token = settings.TELEGRAM_TOKEN
    chat_id = chat_id_destino if chat_id_destino else settings.TELEGRAM_CHAT_ID
    
    origen_etiqueta = "📞 LLAMADA GRABADA DEL SISTEMA" if tipo_origen == "llamada" else "📁 ARCHIVO MULTIMEDIA EXTERNO"
    
    print("\n📡 [DEBUG TELEGRAM] --- INICIANDO CIRCUITO DE ALERTA ---")
    print(f"   -> Token en memoria: {'SÍ (Detectado)' if token else 'NO (Vacío - Revisa el .env)'}")
    print(f"   -> Chat ID Destino: {chat_id if chat_id else 'NO (Vacío)'}")
    print(f"   -> Procedencia: {tipo_origen.upper()}")
    
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            
            # Construcción de la alerta enriquecida con pautas del Motor 4
            mensaje = (
                f"🚨 ALERTA CALLSHIELD - DETECCIÓN DE FRAUDE 🚨\n\n"
                f"📌 Origen: {origen_etiqueta}\n"
                f"📄 Archivo: {archivo}\n"
                f"⚠️ Nivel de Riesgo: {riesgo}\n\n"
                f"🛡️ RECOMENDACIONES INMEDIATAS:\n"
            )
            for rec in recomendaciones:
                mensaje += f"- {rec}\n"
                
            res = requests.post(url, json={"chat_id": chat_id, "text": mensaje}, timeout=5)
            print(f"📥 [DEBUG TELEGRAM] Respuesta de Telegram API: Código HTTP {res.status_code}")
            if res.status_code != 200:
                print(f"   -> Detalle del Fallo de Telegram: {res.text}")
        except Exception as e:
            print(f"❌ [DEBUG TELEGRAM] Excepción crítica de red: {str(e)}")
    else:
        print("⚠️ [DEBUG TELEGRAM] Envío omitido de forma segura: Faltan credenciales Token o Chat ID.")
    print("------------------------------------------------------------\n")


def calcular_riesgo_y_recomendaciones(
    score_voz_ia: float,
    resultado_social: dict
) -> dict:
    """
    Calcula el riesgo global unificado combinando el análisis de biometría acústica (Motor 1)
    y el análisis de ingeniería social/NLP (Motor 3) para archivos multimedia de audio.
    """
    riesgo_social = resultado_social.get("riesgo_social", 0)

    if score_voz_ia >= 70:
        riesgo_global = round(max(score_voz_ia, (score_voz_ia * 0.25) + (riesgo_social * 0.75)))
    else:
        riesgo_global = round((score_voz_ia * 0.25) + (riesgo_social * 0.75))

    if riesgo_global >= 80:
        nivel = "CRÍTICO / ALTO RIESGO"
        recomendaciones = [
            "No continúe la conversación bajo ningún concepto.",
            "No entregue contraseñas, códigos de verificación SMS ni pines bancarios.",
            "Cuelgue de inmediato y contacte a su entidad financiera mediante canales oficiales.",
            "Reporte este número telefónico a las autoridades competentes."
        ]
    elif riesgo_global >= 60:
        nivel = "ALTO / MUY SOSPECHOSO"
        recomendaciones = [
            "Alta probabilidad de suplantación de identidad mediante IA generativa.",
            "Los patrones léxicos indican un vector claro de manipulación o urgencia falsa.",
            "No ejecute acciones financieras ni operativas solicitadas en este fragmento de audio.",
            "Efectúe una verificación de identidad en canales oficiales fuera de banda."
        ]
    elif riesgo_global >= 35:
        nivel = "MEDIO / SOSPECHOSO"
        recomendaciones = [
            "Proceda con precaución analítica. Muestra con anomalías moderadas.",
            "Se detectaron estructuras lingüísticas persuasivas que sugieren precaución.",
            "Evite realizar depósitos o transferencias sin una doble validación con el emisor real."
        ]
    else:
        nivel = "BAJO / RECONOCIMIENTO NORMAL"
        recomendaciones = [
            "Muestra estable. No se detectaron anomalías severas de ingeniería social o clonación.",
            "Conserve las pautas estándar de seguridad digital y monitoreo de medios."
        ]

    return {
        "riesgo_global": riesgo_global,
        "nivel_evaluacion": nivel,
        "recomendaciones": recomendaciones
    }