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
    nivel_confianza_voz: str = None,
    resumen_seguridad: str = ""
):
    token = settings.TELEGRAM_TOKEN
    chat_id = chat_id_destino if chat_id_destino else settings.TELEGRAM_CHAT_ID
    origen_etiqueta = "📱 NOTA DE VOZ DE WHATSAPP" if origen.lower() == "whatsapp" else "📞 LLAMADA GRABADA"
    
    if not token or not chat_id:
        print("⚠️ [Telegram] Omitido: Faltan credenciales en el .env")
        return

    texto_transcrito = transcripcion if len(transcripcion) < 800 else transcripcion[:800] + "... [truncado]"
    tacticas_str = ", ".join(tacticas).replace("_", " ").title() if tacticas else "Ninguna evidente"
    
    biometria_str = ""
    if score_voz_ia is not None:
        biometria_str = (
            f"🔬 ANÁLISIS DE LA VOZ:\n"
            f"• Probabilidad de que sea una voz falsa o creada por computadora: {score_voz_ia} de 100\n"
            f"• Evaluación de la llamada: {nivel_confianza_voz if nivel_confianza_voz else 'Voz normal sin alteraciones'}\n\n"
        )

    resumen_str = ""
    if resumen_seguridad:
        resumen_str = f"💡 RESUMEN RÁPIDO:\n{resumen_seguridad}\n\n"

    mensaje = (
        f"🚨 INFORME DE SEGURIDAD GuardIAn 🚨\n\n"
        f"📂 Canal de comunicación: {origen_etiqueta}\n"
        f"📄 Archivo analizado: {archivo}\n"
        f"⚠️ Nivel de peligro detectado: {riesgo}\n\n"
        f"{resumen_str}"
        f"{biometria_str}"
        f"🗣️ TRANSCRIPCIÓN DEL AUDIO:\n\"{texto_transcrito}\"\n\n"
        f"🧠 COMPORTAMIENTOS SOSPECHOSOS DETECTADOS:\n{tacticas_str}\n\n"
        f"🛡️ RECOMENDACIONES DE PROTECCIÓN INMEDIATA:\n"
    )
    for rec in recomendaciones:
        mensaje += f"• {rec}\n"

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, json={"chat_id": chat_id, "text": mensaje}, timeout=5)
        if res.status_code != 200:
            print(f"❌ [Telegram Error] {res.text}")
        else:
            print("✅ [Telegram] Reporte enviado al usuario con éxito.")
    except Exception as e:
        print(f"❌ [Telegram Error de Red] {str(e)}")


def calcular_riesgo_y_recomendaciones(score_voz_ia: float, resultado_social: dict) -> dict:
    riesgo_social = resultado_social.get("riesgo_social", 0)
    fraude_detectado = resultado_social.get("fraude_detectado", False)
    tacticas = resultado_social.get("tacticas", [])
    es_narracion_tts = resultado_social.get("es_narracion_tts", False)
    alerta_lexico_local = resultado_social.get("alerta_lexico_local", False)
    descripcion_lexico = resultado_social.get("descripcion_lexico", "")
    muestra_legible = resultado_social.get("muestra_legible", True)
    variaciones_idioma = resultado_social.get("variaciones_idioma_detectadas", False)


    es_dialogo_seguro = (riesgo_social < 25 and not fraude_detectado and not es_narracion_tts and not alerta_lexico_local)

    if alerta_lexico_local or not muestra_legible:
        riesgo_social = min(riesgo_social + 20, 100)

    if riesgo_social < 30 and not fraude_detectado:
        if score_voz_ia >= 70:
            if es_narracion_tts:
                riesgo_global = round((score_voz_ia * 0.40) + (riesgo_social * 0.60))
            else:
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
        recomendaciones.append("Se aconseja cortar contacto y proteger de inmediato sus cuentas financieras.")
        resumen_seguridad = "¡Peligro inminente! El audio contiene señales extremas de engaño o extorsión. No envíe dinero ni información."
    elif riesgo_global >= 60:
        nivel = "ALTO / MUY SOSPECHOSO"
        recomendaciones.append("Alta probabilidad de intento de engaño. No realice ninguna acción operativa.")
        resumen_seguridad = "¡Cuidado! Hay altas sospechas de manipulación en esta nota de voz."
    elif riesgo_global >= 35:
        nivel = "MEDIO / SOSPECHOSO"
        recomendaciones.append("Proceda con precaución. Muestra con anomalías moderadas en el discurso o la acústica.")
        if score_voz_ia >= 70:
            resumen_seguridad = "Aviso de Identidad: Se detectó una voz artificial o clonada. Aunque el mensaje no contiene amenazas directas, actúe con precaución."
        else:
            resumen_seguridad = "Tenga precaución. El audio tiene algunos elementos inusuales."
    else:
        nivel = "BAJO / NORMAL"
        recomendaciones.append("Muestra segura y sin rasgos evidentes de peligro.")
        if es_narracion_tts:
            resumen_seguridad = "Aviso: El audio fue generado por una computadora para leer un texto, pero es inofensivo."
        elif score_voz_ia >= 70 and not es_dialogo_seguro:
            resumen_seguridad = "Aviso de Identidad: Se detectó el uso de una voz artificial en un mensaje común. Posible cuenta simulada."
        else:
            resumen_seguridad = "Muestra segura: Es una conversación normal sin alertas de riesgo."

    if score_voz_ia >= 70:
        if es_dialogo_seguro:
            recomendaciones.append("Indicador Acústico: Se detectó una alteración espectral aislada por compresión de WhatsApp, pero el contenido es completamente seguro.")
        else:
            recomendaciones.append("Alerta de Voz Clonada: Existen indicios severos de suplantación artificial (Deepfake).")
            recomendaciones.append("Recomendación: Confirme la identidad del emisor mediante una pregunta privada por un medio alterno.")
    else:
        if es_dialogo_seguro:
            recomendaciones.append("Muestra verificada. No se requiere ninguna acción defensiva inmediata.")

    if fraude_detectado or riesgo_social >= 35:
        recomendaciones.append("Alerta de Contenido: El discurso presenta patrones reales de persuasión maliciosa o coacción.")
        if tacticas:
            tacticas_formateadas = ", ".join(tacticas).replace("_", " ").title()
            recomendaciones.append(f"Indicadores identificados: {tacticas_formateadas}.")
        recomendaciones.append("Medida de Ciberseguridad: Bajo ninguna circunstancia entregue códigos SMS ni realice transferencias.")
    else:
        if riesgo_global < 35 and score_voz_ia < 70:
            recomendaciones.append("Conserve las pautas estándar de cuidado de sus datos personales.")

    if alerta_lexico_local:
        recomendaciones.append(f"Alerta Regional: Se identificaron palabras comúnmente usadas en extorsiones ({descripcion_lexico}).")
    if not muestra_legible:
        recomendaciones.append("Aviso de Distorsión: El audio está severamente dañado o su contenido es ininteligible, lo cual es sospechoso.")
    if variaciones_idioma:
        recomendaciones.append("Aviso Lingüístico: Se detectaron mezclas de idiomas inusuales en la conversación.")

    return {
        "riesgo_global": riesgo_global,
        "nivel_evaluacion": nivel,
        "resumen_seguridad": resumen_seguridad,
        "recomendaciones": recomendaciones
    }