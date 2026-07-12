import os
import requests
from app.config import settings

def enviar_reporte_telegram(
    archivo: str, 
    origen: str, 
    transcripcion: str, 
    tacticas: list, 
    riesgo: str, 
    recomendaciones: list, 
    chat_id_destino: str = None,
    score_voz_ia: float = None,
    nivel_confianza_voz: str = None
):
    token = settings.TELEGRAM_TOKEN
    chat_id = chat_id_destino if chat_id_destino else settings.TELEGRAM_CHAT_ID
    origen_etiqueta = "📱 AUDIO DE WHATSAPP" if origen.lower() == "whatsapp" else "📞 LLAMADA GRABADA"
    
    if not token or not chat_id:
        print("⚠️ [Telegram] Omitido: Faltan credenciales en el .env")
        return

    texto_transcrito = transcripcion if len(transcripcion) < 800 else transcripcion[:800] + "... [truncado]"
    tacticas_str = ", ".join(tacticas).replace("_", " ").title() if tacticas else "Ninguna evidente"
    
    biometria_str = ""
    if score_voz_ia is not None:
        biometria_str = (
            f"🔬 ANÁLISIS BIO-ACÚSTICO (MOTOR 1):\n"
            f"• Score de Voz Sintética/IA: {score_voz_ia}/100\n"
            f"• Diagnóstico Clínico: {nivel_confianza_voz if nivel_confianza_voz else 'N/A'}\n\n"
        )

    mensaje = (
        f"🚨 INFORME PERICIAL CALLSHIELD 🚨\n\n"
        f"📂 Origen: {origen_etiqueta}\n"
        f"📄 Archivo: {archivo}\n"
        f"⚠️ Nivel de Riesgo Consolidado: {riesgo}\n\n"
        f"{biometria_str}"
        f"🗣️ TRANSCRIPCIÓN DETECTADA:\n\"{texto_transcrito}\"\n\n"
        f"🧠 TÁCTICAS DE EXTORSIÓN/MANIPULACIÓN:\n{tacticas_str}\n\n"
        f"🛡️ RECOMENDACIONES PREVENTIVAS:\n"
    )
    for rec in recomendaciones:
        mensaje += f"• {rec}\n"

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, json={"chat_id": chat_id, "text": mensaje}, timeout=5)
        if res.status_code != 200:
            print(f"❌ [Telegram Error] {res.text}")
        else:
            print("✅ [Telegram] Reporte exhaustivo enviado al usuario con éxito.")
    except Exception as e:
        print(f"❌ [Telegram Error de Red] {str(e)}")


def calcular_riesgo_y_recomendaciones(score_voz_ia: float, resultado_social: dict) -> dict:
    riesgo_social = resultado_social.get("riesgo_social", 0)
    fraude_detectado = resultado_social.get("fraude_detectado", False)
    tacticas = resultado_social.get("tacticas", [])
    es_narracion_tts = resultado_social.get("es_narracion_tts", False)

    # Evaluación y cálculo balanceado de riesgo unificado
    if riesgo_social < 30 and not fraude_detectado:
        if score_voz_ia >= 70:
            if es_narracion_tts:
                # Es un bot de lectura real legítimo, elevamos levemente el riesgo preventivo
                riesgo_global = round((score_voz_ia * 0.40) + (riesgo_social * 0.60))
            else:
                # Mensaje benigno con sospecha acústica aislada por codec Opus
                riesgo_global = round((score_voz_ia * 0.25) + (riesgo_social * 0.75))
        else:
            riesgo_global = round((score_voz_ia * 0.25) + (riesgo_social * 0.75))
    else:
        if score_voz_ia >= 70:
            riesgo_global = round(max(score_voz_ia, (score_voz_ia * 0.3) + (riesgo_social * 0.7)))
        else:
            riesgo_global = round((score_voz_ia * 0.25) + (riesgo_social * 0.75))
            if fraude_detectado:
                riesgo_global = max(riesgo_global, 60)

    recomendaciones = []
    
    if riesgo_global >= 80:
        nivel = "CRÍTICO / ALTO RIESGO"
        recomendaciones.append("Evidencia de alto peligro extremo. Se recomienda el aislamiento inmediato del archivo de audio.")
    elif riesgo_global >= 60:
        nivel = "ALTO / MUY SOSPECHOSO"
        recomendaciones.append("Alta probabilidad de intento de manipulación psicológica, coacción o suplantación activa.")
    elif riesgo_global >= 35:
        nivel = "MEDIO / SOSPECHOSO"
        recomendaciones.append("Proceda con precaución analítica. Muestra con anomalías moderadas en el discurso o la acústica.")
    else:
        nivel = "BAJO / NORMAL"
        if score_voz_ia >= 70 and not es_narracion_tts:
            recomendaciones.append("Muestra lingüísticamente estable y sin rasgos de fraude, bajo observación preventiva por indicadores acústicos.")
        else:
            recomendaciones.append("Muestra acústicamente estable y sin rasgos evidentes de fraude o ingeniería social.")

    # Recomendaciones analíticas basadas en el Motor 1 (Acústico) y la nueva bandera TTS
    if score_voz_ia >= 70:
        if riesgo_social < 30 and not fraude_detectado:
            if es_narracion_tts:
                recomendaciones.append("Identificación Forense: El contexto del texto coincide con locuciones automáticas generadas por software de Texto a Voz (TTS) o asistentes de lectura artificiales.")
            else:
                recomendaciones.append("Aviso Técnico: Se detectó un score acústico elevado en un mensaje inofensivo. Es altamente probable que corresponda a un falso positivo causado por los artefactos de compresión del códec de WhatsApp (Opus).")
        else:
            recomendaciones.append("Alerta de Biometría: Existen indicios severos de clonación de voz o generación por IA (Deepfake). No valide la identidad del emisor únicamente por cómo suena su voz.")
            recomendaciones.append("Acción Sugerida: Realice una pregunta de validación de doble factor (un secreto familiar o dato privado) fuera del canal de WhatsApp.")

    # Recomendaciones analíticas basadas en el Motor 3 (Semántico de OpenAI)
    if fraude_detectado or riesgo_social >= 35:
        recomendaciones.append("Alerta Semántica: El análisis lingüístico identificó patrones reales de coacción o persuasión maliciosa.")
        if tacticas:
            tacticas_formateadas = ", ".join(tacticas).replace("_", " ").title()
            recomendaciones.append(f"Tácticas detectadas por IA: {tacticas_formateadas}.")
        recomendaciones.append("Medida Preventiva: Bajo ninguna circunstancia entregue contraseñas, códigos de verificación SMS o realice transferencias bancarias.")
        recomendaciones.append("Acción Inmediata: Corte la comunicación, bloquee el contacto sospechoso y reporte el número telefónico a las autoridades periciales.")
    else:
        if riesgo_global < 35 and not es_narracion_tts:
            recomendaciones.append("Mantenga las pautas estándar de ciberseguridad y protección de datos personales en redes sociales.")

    return {
        "riesgo_global": riesgo_global,
        "nivel_evaluacion": nivel,
        "recomendaciones": recomendaciones
    }