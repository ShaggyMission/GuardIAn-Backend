import os
import dataclasses
from typing import Optional
import numpy as np
import librosa

from .inference import predecir_segmentos, UMBRAL_DECISION, SAMPLE_RATE, MAX_LEN_MUESTRAS
from .benford_features import analizar_benford_dct

@dataclasses.dataclass
class EvidenciaBenford:
    mad: float
    p_value: float
    categoria_conformidad: str
    n_muestras: int

@dataclasses.dataclass
class EvidenciaEntropia:
    valor_bits: float
    z_score: Optional[float] = None

@dataclasses.dataclass
class EvidenciaNeuronal:
    disponible: bool
    score_fake_pct: Optional[float]
    nombre_modelo: str

@dataclasses.dataclass
class ReporteForense:
    archivo: str
    evidencia_neuronal: EvidenciaNeuronal
    evidencia_benford: EvidenciaBenford
    evidencia_entropia: EvidenciaEntropia
    score_riesgo: float
    nivel_confianza: str
    advertencia: str = (
        "Este resultado es un score de riesgo estadístico de screening, "
        "obtenido combinando la probabilidad del modelo neuronal y análisis estadísticos. "
        "No constituye prueba pericial certificada."
    )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def analizar_entropia_espectral(y: np.ndarray) -> EvidenciaEntropia:
    """
    Calcula la entropía de Shannon sobre la potencia espectral temporal.
    Sirve para medir la uniformidad del espectro del audio.
    """
    espectrograma = np.abs(librosa.stft(y))
    potencia = np.sum(espectrograma ** 2, axis=0)
    suma = np.sum(potencia)
    if suma == 0:
        return EvidenciaEntropia(valor_bits=0.0)

    probs = potencia / suma
    probs = probs[probs > 0]
    entropia = float(-np.sum(probs * np.log2(probs)))
    return EvidenciaEntropia(valor_bits=entropia)

class VoiceAIEngine:
    def __init__(self):
        self.nombre_modelo = "facebook/wav2vec2-xls-r-300m + SLS"
        self.mad_umbral_no_conforme = 0.012
        self.entropia_umbral_bits = 11.5
        
        # Pesos de fusión para el score de riesgo global.
        # Dado que el modelo neuronal SLS está calibrado y validado con F1 ~0.94,
        # le otorgamos el mayor peso. Benford y Entropía actúan como señales auxiliares.
        self.pesos_fusion = {
            "neuronal": 0.8,
            "benford": 0.1,
            "entropia": 0.1
        }

    def analizar_clonacion(self, ruta_audio: str) -> ReporteForense:
        """
        Orquesta el pipeline forense completo:
          1. Predicción con SLSModel (segmentado para soportar audios de >4s).
          2. Análisis de distribución del primer dígito (Ley de Benford en DCT log-espectro).
          3. Análisis de uniformidad/entropía espectral.
          4. Fusión de métricas en un score de riesgo consolidado [0, 100].
        """
        if not os.path.exists(ruta_audio):
            raise FileNotFoundError(f"No se encontró el archivo de audio en: {ruta_audio}")

        # Cargar audio a 16000Hz (mono) para procesamiento general
        y, _ = librosa.load(ruta_audio, sr=SAMPLE_RATE)
        if len(y) == 0:
            raise ValueError("El archivo de audio está vacío o corrupto.")

        # 1. Ejecutar predicción del modelo neuronal segmentada
        res_neuronal = predecir_segmentos(ruta_audio, sr=SAMPLE_RATE, max_len=MAX_LEN_MUESTRAS)
        
        ev_neuronal = EvidenciaNeuronal(
            disponible=True,
            score_fake_pct=res_neuronal["score_fake_pct"],
            nombre_modelo=self.nombre_modelo
        )

        # 2. Análisis Ley de Benford
        res_benford = analizar_benford_dct(y)
        ev_benford = EvidenciaBenford(
            mad=res_benford["mad"],
            p_value=res_benford["p_value"],
            categoria_conformidad=res_benford["categoria_conformidad"],
            n_muestras=res_benford["n_muestras"]
        )

        # 3. Análisis de Entropía Espectral
        ev_entropia = analizar_entropia_espectral(y)

        # --- Fusión y Normalización a [0, 1] de cada componente ---
        
        # Riesgo Neuronal (ya mapeado y calibrado a [0, 1] desde score_fake_pct)
        riesgo_neuronal = ev_neuronal.score_fake_pct / 100.0

        # Riesgo Benford (normalizado por el umbral de no conformidad)
        if ev_benford.categoria_conformidad != "Muestra insuficiente":
            riesgo_benford = min(ev_benford.mad / self.mad_umbral_no_conforme, 1.0)
        else:
            riesgo_benford = 0.5  # Neutro si no hay muestras suficientes

        # Riesgo Entropía (baja entropía indica anomalías o silencios sospechosos)
        # Si valor_bits es menor al umbral, riesgo es 1.0; de lo contrario es 0.0
        riesgo_entropia = 1.0 if ev_entropia.valor_bits < self.entropia_umbral_bits else 0.0

        # Calcular score consolidado
        score_riesgo = 100 * (
            self.pesos_fusion["neuronal"] * riesgo_neuronal +
            self.pesos_fusion["benford"] * riesgo_benford +
            self.pesos_fusion["entropia"] * riesgo_entropia
        )
        score_riesgo = float(np.clip(score_riesgo, 0.0, 100.0))

        # Determinar el nivel de riesgo global
        if score_riesgo >= 80:
            nivel = "Riesgo alto — múltiples señales convergentes"
        elif score_riesgo >= 55:
            nivel = "Riesgo moderado — señales mixtas, requiere revisión"
        elif score_riesgo >= 30:
            nivel = "Riesgo bajo — anomalías menores aisladas"
        else:
            nivel = "Riesgo mínimo — consistente con audio orgánico"

        return ReporteForense(
            archivo=os.path.basename(ruta_audio),
            evidencia_neuronal=ev_neuronal,
            evidencia_benford=ev_benford,
            evidencia_entropia=ev_entropia,
            score_riesgo=round(score_riesgo, 1),
            nivel_confianza=nivel
        )

# Instancia global exportada
voice_ai_engine = VoiceAIEngine()
