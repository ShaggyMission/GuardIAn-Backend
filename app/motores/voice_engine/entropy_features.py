"""
entropy_features.py

Calcula la entropía de Shannon del espectro de un audio.

Por qué esta métrica: la entropía mide qué tan "impredecible" o
distribuida está la energía del audio a lo largo de las distintas
frecuencias. Una señal muy uniforme/artificial (poca variación,
patrones muy regulares) tiende a tener entropía más baja que una
señal con la variabilidad natural de una voz humana grabada en
condiciones reales. Es una métrica descriptiva estándar de
teoría de la información, no específica de detección de deepfakes,
así que —igual que con Benford— conviene tratarla como una señal
complementaria y no como un clasificador por sí sola.

NOTA: igual que benford_features.py, esta métrica no ha sido validada
empíricamente como aporte real al modelo. Usar `validar_aporte_entropia`
antes de integrarla al endpoint final.
"""

import numpy as np
import librosa


def _entropia_shannon(distribucion, base=2):
    """
    Calcula la entropía de Shannon de una distribución de probabilidad.
    `distribucion` debe ser un arreglo de valores no negativos (no hace
    falta que ya esté normalizado, se normaliza aquí).
    """
    distribucion = np.asarray(distribucion, dtype=np.float64)
    distribucion = distribucion[distribucion > 0]  # log(0) no está definido

    if len(distribucion) == 0:
        return 0.0

    probabilidades = distribucion / distribucion.sum()
    return float(-np.sum(probabilidades * np.log(probabilidades) / np.log(base)))


def entropia_espectral(ruta_audio, sr=16000):
    """
    Entropía de Shannon (en bits) de la energía promedio por frecuencia
    del espectrograma del audio completo.
    """
    y, _ = librosa.load(ruta_audio, sr=sr, mono=True)
    stft = np.abs(librosa.stft(y))
    energia_por_frecuencia = np.mean(stft, axis=1)  # promedio a través del tiempo
    return _entropia_shannon(energia_por_frecuencia)


def entropia_por_frame(ruta_audio, sr=16000):
    """
    Entropía de Shannon calculada frame por frame (en vez de promediar
    primero), y luego se agregan estadísticas sobre esas entropías.
    Da una visión de la variabilidad temporal, no solo el promedio.

    Devuelve:
        {
            "entropia_media": float,
            "entropia_std": float,       # qué tanto varía en el tiempo
            "entropia_min": float,
            "entropia_max": float,
        }
    """
    y, _ = librosa.load(ruta_audio, sr=sr, mono=True)
    stft = np.abs(librosa.stft(y))  # shape: (frecuencias, frames_de_tiempo)

    entropias_por_frame = [
        _entropia_shannon(stft[:, i]) for i in range(stft.shape[1])
    ]

    if not entropias_por_frame:
        return {"entropia_media": 0.0, "entropia_std": 0.0, "entropia_min": 0.0, "entropia_max": 0.0}

    entropias_por_frame = np.array(entropias_por_frame)
    return {
        "entropia_media": float(np.mean(entropias_por_frame)),
        "entropia_std": float(np.std(entropias_por_frame)),
        "entropia_min": float(np.min(entropias_por_frame)),
        "entropia_max": float(np.max(entropias_por_frame)),
    }


def analizar_entropia(ruta_audio, sr=16000):
    """
    Resumen listo para incluir en el reporte forense
    (campo `evidencia_entropia.valor_bits` esperado por main.py).
    """
    valor_bits = entropia_espectral(ruta_audio, sr=sr)
    detalle = entropia_por_frame(ruta_audio, sr=sr)

    return {
        "valor_bits": valor_bits,
        **detalle,
    }


# --------------------------------------------------------------------------
# Validación empírica (mismo patrón que benford_features.py)
# --------------------------------------------------------------------------
def validar_aporte_entropia(muestras_validacion, obtener_p_real_modelo):
    """
    Igual que `validar_aporte_benford`: compara F1 usando solo el modelo
    neuronal contra modelo + entropía, para decidir con datos si vale la
    pena integrar esta señal al endpoint final.

    Parámetros:
        muestras_validacion: lista de dicts con {"path": ruta, "label": 1|0}
        obtener_p_real_modelo: función(ruta) -> probabilidad de "real" del modelo
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_solo_modelo = []
    X_con_entropia = []
    y_true = []

    for muestra in muestras_validacion:
        ruta = muestra["path"]
        etiqueta = muestra["label"]

        p_real = obtener_p_real_modelo(ruta)
        entropia = analizar_entropia(ruta)

        X_solo_modelo.append([p_real])
        X_con_entropia.append([p_real, entropia["valor_bits"], entropia["entropia_std"]])
        y_true.append(etiqueta)

    X_solo_modelo = np.array(X_solo_modelo)
    X_con_entropia = np.array(X_con_entropia)
    y_true = np.array(y_true)

    Xs_train, Xs_test, y_train, y_test = train_test_split(
        X_solo_modelo, y_true, test_size=0.3, random_state=42, stratify=y_true
    )
    Xe_train, Xe_test, _, _ = train_test_split(
        X_con_entropia, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    clf_solo = LogisticRegression().fit(Xs_train, y_train)
    f1_solo = f1_score(y_test, clf_solo.predict(Xs_test))

    clf_entropia = LogisticRegression().fit(Xe_train, y_train)
    f1_entropia = f1_score(y_test, clf_entropia.predict(Xe_test))

    mejora = f1_entropia - f1_solo

    print(f"F1 usando solo el modelo neuronal:        {f1_solo:.4f}")
    print(f"F1 usando modelo neuronal + entropía:      {f1_entropia:.4f}")
    print(f"Diferencia:                                {mejora:+.4f}")

    if mejora > 0.01:
        print("\n→ La entropía parece aportar una mejora real. Vale la pena integrarla.")
    else:
        print("\n→ La entropía NO muestra una mejora clara. No se recomienda integrarla "
              "al endpoint final tal como está.")

    return {
        "f1_solo_modelo": f1_solo,
        "f1_con_entropia": f1_entropia,
        "mejora": mejora,
        "recomendado": mejora > 0.01,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python entropy_features.py <ruta_al_audio>")
        sys.exit(1)

    resultado = analizar_entropia(sys.argv[1])
    print(f"\nArchivo: {sys.argv[1]}")
    for clave, valor in resultado.items():
        print(f"{clave}: {valor:.4f}" if isinstance(valor, float) else f"{clave}: {valor}")