import os
from app.config import settings


def calcular_riesgo_y_recomendaciones(
    score_voz_ia: float,
    resultado_social: dict
) -> dict:


    # Obtener riesgo generado por Motor 3
    riesgo_social = resultado_social.get(
        "riesgo_social",
        0
    )


    # Riesgo consolidado
    riesgo_global = round(
        (score_voz_ia * 0.25)
        +
        (riesgo_social * 0.75)
    )


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
            "Existe una alta probabilidad de intento de fraude.",
            "No realice transferencias ni envíe dinero sin verificar la identidad por otro medio.",
            "Confirme la información llamando directamente a la persona o institución."
    ]

    elif riesgo_global >= 35:
        nivel = "MEDIO / SOSPECHOSO"
        recomendaciones = [
            "Proceda con extrema precaución.",
            "No realice transferencias ni depósitos sin verificar la identidad.",
            "Verifique la información mediante otro canal."
    ]

    else:
        nivel = "BAJO / RECONOCIMIENTO NORMAL"
        recomendaciones = [
            "No se detectaron anomalías severas de ingeniería social o clonación.",
            "Manténgase atento ante solicitudes inusuales."
    ]


    return {

        "riesgo_global": riesgo_global,

        "nivel_evaluacion": nivel,

        "recomendaciones": recomendaciones
    }