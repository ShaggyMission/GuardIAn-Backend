"""
inference.py

Carga el modelo SLS (XLS-R + cabeza de atención) ya afinado y expone
funciones para predecir si un audio contiene voz real o generada por IA.

Requiere en la misma carpeta:
    - model.py                 (clase SLSModel)
    - sls_best.pth              (checkpoint del fine-tuning)
    - config_produccion.json    (umbral y parámetros de preprocesamiento)
"""

import os
import json
import numpy as np
import torch
import librosa

from model import SLSModel

# --------------------------------------------------------------------------
# Rutas y configuración
# --------------------------------------------------------------------------
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_CONFIG = os.path.join(DIR_ACTUAL, "config_produccion.json")
RUTA_CHECKPOINT = os.path.join(DIR_ACTUAL, "sls_best.pth")

with open(RUTA_CONFIG, "r") as f:
    CONFIG = json.load(f)

SAMPLE_RATE = CONFIG["sample_rate_entrada"]      # 16000
MAX_LEN = CONFIG["max_len_muestras"]             # 64600 (~4 segundos)
UMBRAL_DECISION = CONFIG["umbral_decision"]      # 0.999979

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# Carga del modelo (una sola vez, al importar este módulo)
# --------------------------------------------------------------------------
_modelo = None


def _cargar_modelo():
    """Carga el modelo en memoria una sola vez (patrón singleton simple)."""
    global _modelo
    if _modelo is not None:
        return _modelo

    if not os.path.exists(RUTA_CHECKPOINT):
        raise FileNotFoundError(
            f"No se encontró el checkpoint del modelo en: {RUTA_CHECKPOINT}\n"
            "Asegúrate de haber descargado 'sls_best.pth' desde Kaggle y "
            "colocarlo en esta misma carpeta."
        )

    modelo = SLSModel(DEVICE).to(DEVICE)

    state_dict = torch.load(RUTA_CHECKPOINT, map_location=DEVICE)
    # Por si el checkpoint se guardó bajo DataParallel (prefijo "module.")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict)

    modelo.eval()
    _modelo = modelo
    return _modelo


# --------------------------------------------------------------------------
# Preprocesamiento de audio
# --------------------------------------------------------------------------
def _pad(x, max_len=MAX_LEN):
    """Recorta o repite el audio hasta llegar exactamente a max_len muestras."""
    if len(x) >= max_len:
        return x[:max_len]
    repeticiones = int(max_len / len(x)) + 1
    return np.tile(x, repeticiones)[:max_len]


def _cargar_audio(ruta_audio, sr=SAMPLE_RATE):
    """
    Carga cualquier formato soportado por librosa/ffmpeg (wav, mp3, opus, m4a, ogg...)
    y lo resamplea al sample rate que el modelo espera.
    """
    y, _ = librosa.load(ruta_audio, sr=sr, mono=True)
    return y


# --------------------------------------------------------------------------
# Predicción
# --------------------------------------------------------------------------
def predecir(ruta_audio, umbral=None):
    """
    Analiza un archivo de audio completo (solo sus primeros ~4 segundos,
    ver `predecir_segmentado` para audios largos) y determina si la voz
    es real o generada por IA.

    Devuelve:
        {
            "es_voz_ia": bool,
            "confianza_real": float,   # probabilidad de que sea voz real (0-1)
            "confianza_ia": float,     # probabilidad de que sea voz IA (0-1)
            "umbral_usado": float
        }
    """
    if umbral is None:
        umbral = UMBRAL_DECISION

    modelo = _cargar_modelo()

    y = _cargar_audio(ruta_audio)
    y = _pad(y).astype(np.float32)
    x = torch.tensor(y).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        salida = modelo(x)
        p_real = torch.exp(salida[0][1]).item()
        p_falso = torch.exp(salida[0][0]).item()

    return {
        "es_voz_ia": p_real < umbral,
        "confianza_real": p_real,
        "confianza_ia": p_falso,
        "umbral_usado": umbral,
    }


def predecir_segmentado(ruta_audio, umbral=None, duracion_segmento_s=4.0):
    """
    Para audios más largos que ~4 segundos: divide el audio en segmentos
    consecutivos de esa duración, analiza cada uno por separado, y agrega
    los resultados (se marca como IA si ALGÚN segmento se detecta como IA,
    ya que basta con que una parte del audio sea sintética para que el
    mensaje completo sea sospechoso).

    Útil porque el modelo internamente solo analiza los primeros 4 segundos
    de cualquier archivo que se le pase directo — esta función evita perder
    información si el contenido relevante está más adelante en el audio.
    """
    if umbral is None:
        umbral = UMBRAL_DECISION

    modelo = _cargar_modelo()
    y_completo = _cargar_audio(ruta_audio)

    muestras_por_segmento = int(duracion_segmento_s * SAMPLE_RATE)
    total_muestras = len(y_completo)

    if total_muestras <= muestras_por_segmento:
        resultado = predecir(ruta_audio, umbral=umbral)
        resultado["segmentos"] = [resultado.copy()]
        return resultado

    resultados_segmentos = []
    for inicio in range(0, total_muestras, muestras_por_segmento):
        fin = inicio + muestras_por_segmento
        segmento = y_completo[inicio:fin]

        # Descarta segmentos finales demasiado cortos (menos de 1 segundo)
        if len(segmento) < SAMPLE_RATE:
            continue

        segmento_pad = _pad(segmento, MAX_LEN).astype(np.float32)
        x = torch.tensor(segmento_pad).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            salida = modelo(x)
            p_real = torch.exp(salida[0][1]).item()
            p_falso = torch.exp(salida[0][0]).item()

        resultados_segmentos.append({
            "es_voz_ia": p_real < umbral,
            "confianza_real": p_real,
            "confianza_ia": p_falso,
            "umbral_usado": umbral,
            "inicio_s": inicio / SAMPLE_RATE,
            "fin_s": fin / SAMPLE_RATE,
        })

    if not resultados_segmentos:
        # Audio demasiado corto incluso para un segmento válido
        return predecir(ruta_audio, umbral=umbral)

    algun_segmento_es_ia = any(s["es_voz_ia"] for s in resultados_segmentos)
    peor_confianza_real = min(s["confianza_real"] for s in resultados_segmentos)

    return {
        "es_voz_ia": algun_segmento_es_ia,
        "confianza_real": peor_confianza_real,
        "confianza_ia": max(s["confianza_ia"] for s in resultados_segmentos),
        "umbral_usado": umbral,
        "segmentos": resultados_segmentos,
    }


# --------------------------------------------------------------------------
# Prueba rápida manual (python inference.py ruta/al/audio.wav)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python inference.py <ruta_al_audio>")
        sys.exit(1)

    ruta = sys.argv[1]
    resultado = predecir_segmentado(ruta)

    print(f"\nArchivo: {ruta}")
    print(f"Veredicto: {'VOZ IA / FALSA' if resultado['es_voz_ia'] else 'VOZ REAL'}")
    print(f"Confianza real (peor segmento): {resultado['confianza_real']:.6f}")
    print(f"Umbral usado: {resultado['umbral_usado']}")
    if len(resultado.get("segmentos", [])) > 1:
        print(f"\nDesglose por segmento ({len(resultado['segmentos'])} segmentos):")
        for i, seg in enumerate(resultado["segmentos"]):
            veredicto = "IA" if seg["es_voz_ia"] else "real"
            print(f"  [{seg['inicio_s']:.1f}s - {seg['fin_s']:.1f}s] {veredicto} (p_real={seg['confianza_real']:.6f})")