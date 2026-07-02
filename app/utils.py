import os
from app.config import settings

def calcular_riesgo_y_recomendaciones(score_voz_ia: int, tacticas: dict) -> dict:
    """
    Motor 4: Consolida los puntajes de los motores y genera el veredicto forense.
    Ponderación: 40% Detección de Voz IA + 60% Contenido de Ingeniería Social.
    """
    # Buscamos la táctica de ingeniería social con el puntaje más alto
    max_score_social = max(tacticas.values()) if tacticas else 0
    
    # Cálculo del riesgo global ponderado (Fase 2-3 híbrida)
    riesgo_global = round((score_voz_ia * 0.40) + (max_score_social * 0.60))
    
    # Determinar el nivel cualitativo
    if riesgo_global >= settings.UMBRAL_RIESGO_CRITICO:
        nivel = "CRÍTICO / ALTO RIESGO"
        recomendaciones = [
            "No continúe la conversación bajo ningún concepto.",
            "No entregue contraseñas, códigos de verificación SMS ni pines bancarios.",
            "Cuelgue de inmediato y contacte a su entidad financiera mediante canales oficiales.",
            "Reporte este número telefónico a las autoridades competentes."
        ]
    elif riesgo_global >= 45:
        nivel = "MEDIO / SOSPECHOSO"
        recomendaciones = [
            "Proceda con extrema precaución.",
            "No realice transferencias ni depósitos sin verificar la identidad por un canal alterno.",
            "Haga preguntas de seguridad que solo la persona real conocería."
        ]
    else:
        nivel = "BAJO / RECONOCIMIENTO NORMAL"
        recomendaciones = [
            "No se detectaron anomalías severas de ingeniería social o clonación.",
            "Manténgase siempre atento a solicitudes inusuales de dinero."
        ]
        
    return {
        "riesgo_global": riesgo_global,
        "nivel_evaluacion": nivel,
        "recomendaciones": recomendaciones
    }