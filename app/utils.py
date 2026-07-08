import os
from app.config import settings


def calcular_riesgo_y_recomendaciones(
    score_voz_ia: float,
    resultado_social: dict
) -> dict:
    """
    Calcula el riesgo global unificado combinando el análisis de biometría acústica (Motor 1)
    y el análisis de ingeniería social/NLP (Motor 3) para archivos multimedia de audio.
    """

    riesgo_social = resultado_social.get(
        "riesgo_social",
        0
    )

    if score_voz_ia >= 70:
        riesgo_global = round(max(score_voz_ia, (score_voz_ia * 0.25) + (riesgo_social * 0.75)))
    else:
        riesgo_global = round(
            (score_voz_ia * 0.25)
            +
            (riesgo_social * 0.75)
        )

    if riesgo_global >= 80:
        nivel = "CRÍTICO / ALTO RIESGO"
        recomendaciones = [
            "Evidencia altamente peligrosa. Se recomienda el aislamiento inmediato del archivo.",
            "Contenido con rasgos extremos de clonación de voz neuronal (Deepfake) y coacción.",
            "No valide transferencias, pagos ni divulgación de credenciales basados en este audio.",
            "Exporte este informe pericial y repórtelo a los canales de seguridad correspondientes."
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
        "nivel": nivel,  
        "recomendaciones_seguridad": recomendaciones  
        }