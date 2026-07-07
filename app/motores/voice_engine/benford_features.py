"""
benford_features.py

Módulo EXPERIMENTAL. Calcula qué tan bien los coeficientes MFCC y las
magnitudes del espectrograma de un audio se ajustan a la distribución
de primeros dígitos esperada por la Ley de Benford.

Hipótesis (no confirmada en literatura de detección de deepfakes de audio,
es una analogía tomada de forense de imágenes JPEG, donde SÍ está validado
usar Benford sobre coeficientes DCT para detectar recompresión/manipulación):
la voz sintética podría desviarse de forma distinta a la voz real de la
distribución esperada, al tratarse de coeficientes de transformadas
(MFCC viene de una DCT, el espectrograma de una FFT).

IMPORTANTE: antes de usar el resultado de este módulo como parte del
score final del sistema, hay que validar empíricamente si aporta alguna
mejora real (ver `validar_aporte_benford` al final de este archivo).
Si no la aporta, lo correcto es no integrarlo al endpoint final.
"""

import numpy as np
import librosa


# --------------------------------------------------------------------------
# Núcleo: análisis de primer dígito y comparación contra Benford
# --------------------------------------------------------------------------
def _primer_digito(valores):
    """
    Extrae el primer dígito significativo (1-9) de cada valor absoluto
    distinto de cero en el arreglo.
    """
    valores = np.abs(valores[valores != 0])
    if len(valores) == 0:
        return np.array([])

    exponentes = np.floor(np.log10(valores))
    primer_digito = (valores / (10 ** exponentes)).astype(int)

    # Protección: por redondeo en casos límite, algún valor puede caer fuera de 1-9
    return primer_digito[(primer_digito >= 1) & (primer_digito <= 9)]


def _distribucion_benford_teorica():
    """La distribución esperada de primeros dígitos según la Ley de Benford."""
    return np.array([np.log10(1 + 1 / d) for d in range(1, 10)])


def _mad_contra_benford(digitos_observados):
    """
    Mean Absolute Deviation (MAD) entre la distribución observada de primeros
    dígitos y la distribución teórica de Benford. Es la métrica estándar
    usada en auditoría forense contable para medir conformidad con Benford.

    Valores más altos = mayor desviación de lo esperado por Benford.
    Devuelve None si no hay suficientes datos para calcular la distribución.
    """
    if len(digitos_observados) < 30:  # muestra mínima razonable
        return None

    conteo = np.array([np.sum(digitos_observados == d) for d in range(1, 10)])
    distribucion_obs = conteo / conteo.sum()
    distribucion_teo = _distribucion_benford_teorica()

    return float(np.mean(np.abs(distribucion_obs - distribucion_teo)))


# Umbrales de referencia usados en auditoría forense (Nigrini, 2012) para
# clasificar qué tan "conforme" es una distribución con Benford.
# Se reutilizan aquí solo como una categorización descriptiva, no como
# un criterio de clasificación real/falso validado para audio.
def _categorizar_mad(mad):
    if mad is None:
        return "sin_datos_suficientes"
    if mad < 0.006:
        return "conformidad_cercana"
    if mad < 0.012:
        return "conformidad_aceptable"
    if mad < 0.015:
        return "conformidad_marginal"
    return "no_conforme"


# --------------------------------------------------------------------------
# Extracción de features desde audio
# --------------------------------------------------------------------------
def score_benford_mfcc(ruta_audio, sr=16000, n_mfcc=20):
    """MAD de Benford calculado sobre los coeficientes MFCC del audio."""
    y, _ = librosa.load(ruta_audio, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return _mad_contra_benford(_primer_digito(mfcc.flatten()))


def score_benford_espectrograma(ruta_audio, sr=16000):
    """MAD de Benford calculado sobre las magnitudes del espectrograma (STFT)."""
    y, _ = librosa.load(ruta_audio, sr=sr, mono=True)
    stft = np.abs(librosa.stft(y))
    return _mad_contra_benford(_primer_digito(stft.flatten()))


def analizar_benford(ruta_audio, sr=16000):
    """
    Calcula ambos scores de Benford (MFCC y espectrograma) para un audio,
    y devuelve un resumen listo para incluir en el reporte forense.

    Devuelve:
        {
            "mad_mfcc": float | None,
            "mad_espectrograma": float | None,
            "mad": float | None,              # promedio de ambos, si están disponibles
            "categoria_conformidad": str,
        }
    """
    y, _ = librosa.load(ruta_audio, sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mad_mfcc = _mad_contra_benford(_primer_digito(mfcc.flatten()))

    stft = np.abs(librosa.stft(y))
    mad_espectro = _mad_contra_benford(_primer_digito(stft.flatten()))

    valores_disponibles = [v for v in [mad_mfcc, mad_espectro] if v is not None]
    mad_promedio = float(np.mean(valores_disponibles)) if valores_disponibles else None

    return {
        "mad_mfcc": mad_mfcc,
        "mad_espectrograma": mad_espectro,
        "mad": mad_promedio,
        "categoria_conformidad": _categorizar_mad(mad_promedio),
    }


# --------------------------------------------------------------------------
# Validación empírica (correr esto ANTES de confiar en el módulo)
# --------------------------------------------------------------------------
def validar_aporte_benford(muestras_validacion, obtener_p_real_modelo):
    """
    Evalúa si añadir los scores de Benford a la probabilidad del modelo
    neuronal mejora la separación entre audios reales y falsos, comparado
    con usar solo el modelo neuronal.

    Parámetros:
        muestras_validacion: lista de dicts, cada uno con al menos
            {"path": ruta_al_audio, "label": 1 si es real / 0 si es falso}
        obtener_p_real_modelo: función que recibe una ruta de audio y
            devuelve la probabilidad de "real" del modelo neuronal
            (por ejemplo, envolver `inference.predecir` y devolver
            resultado["confianza_real"])

    Imprime la comparación de F1 (solo modelo vs modelo+Benford) y
    devuelve el reporte en un dict, para decidir si vale la pena
    integrar Benford al endpoint final.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_solo_modelo = []
    X_con_benford = []
    y_true = []

    for muestra in muestras_validacion:
        ruta = muestra["path"]
        etiqueta = muestra["label"]

        p_real = obtener_p_real_modelo(ruta)
        benford = analizar_benford(ruta)

        mad_mfcc = benford["mad_mfcc"] if benford["mad_mfcc"] is not None else 0.0
        mad_espectro = benford["mad_espectrograma"] if benford["mad_espectrograma"] is not None else 0.0

        X_solo_modelo.append([p_real])
        X_con_benford.append([p_real, mad_mfcc, mad_espectro])
        y_true.append(etiqueta)

    X_solo_modelo = np.array(X_solo_modelo)
    X_con_benford = np.array(X_con_benford)
    y_true = np.array(y_true)

    Xs_train, Xs_test, y_train, y_test = train_test_split(
        X_solo_modelo, y_true, test_size=0.3, random_state=42, stratify=y_true
    )
    Xc_train, Xc_test, _, _ = train_test_split(
        X_con_benford, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    clf_solo = LogisticRegression().fit(Xs_train, y_train)
    f1_solo = f1_score(y_test, clf_solo.predict(Xs_test))

    clf_benford = LogisticRegression().fit(Xc_train, y_train)
    f1_benford = f1_score(y_test, clf_benford.predict(Xc_test))

    mejora = f1_benford - f1_solo

    print(f"F1 usando solo el modelo neuronal:      {f1_solo:.4f}")
    print(f"F1 usando modelo neuronal + Benford:     {f1_benford:.4f}")
    print(f"Diferencia:                              {mejora:+.4f}")

    if mejora > 0.01:
        print("\n→ Benford parece aportar una mejora real. Vale la pena integrarlo.")
    else:
        print("\n→ Benford NO muestra una mejora clara. No se recomienda integrarlo "
              "al endpoint final tal como está.")

    return {
        "f1_solo_modelo": f1_solo,
        "f1_con_benford": f1_benford,
        "mejora": mejora,
        "recomendado": mejora > 0.01,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python benford_features.py <ruta_al_audio>")
        sys.exit(1)

    resultado = analizar_benford(sys.argv[1])
    print(f"\nArchivo: {sys.argv[1]}")
    print(f"MAD (MFCC):          {resultado['mad_mfcc']}")
    print(f"MAD (Espectrograma): {resultado['mad_espectrograma']}")
    print(f"MAD promedio:        {resultado['mad']}")
    print(f"Categoría:           {resultado['categoria_conformidad']}")