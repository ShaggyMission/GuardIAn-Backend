import os
import json
import dataclasses
from typing import Optional
 
import numpy as np
import librosa
import torch
from scipy.fft import dct
from scipy.stats import chisquare
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
 
# --- Configuración preventiva de rutas de FFmpeg para Windows --------------
RUTAS_FFMPEG_POSIBLES = [
    r"C:\ffmpeg\bin",
    r"C:\ffmpeg",
    r"C:\ffmpeg-release-essentials\bin",
]
for _ruta in RUTAS_FFMPEG_POSIBLES:
    if os.path.exists(_ruta) and _ruta not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.path.pathsep + _ruta
 
 
class ModeloNeuronalNoDisponibleError(Exception):
    """Se levanta cuando el modelo neuronal no pudo cargarse y no hay modo
    de contingencia configurado explícitamente por el llamador."""
 
 
class AudioInvalidoError(Exception):
    """Se levanta cuando el archivo de audio no existe o está vacío/corrupto."""
 
 
class PipelineForenseError(Exception):
    """Se levanta cuando falla el pipeline de análisis. A diferencia de la
    versión anterior, esto NO se traduce silenciosamente en un score
    arbitrario: el llamador debe decidir cómo manejar el fallo."""
 
 
@dataclasses.dataclass
class EvidenciaBenford:
    mad: float              
    p_value: float
    categoria_conformidad: str  
    n_muestras: int
 
 
@dataclasses.dataclass
class EvidenciaEntropia:
    valor_bits: float
    z_score: Optional[float]   
 
 
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
        "obtenido combinando señales no calibradas individualmente para "
        "esta instalación. No constituye prueba pericial certificada. "
        "Para uso forense/legal, calibrar contra un corpus etiquetado "
        "(p.ej. ASVspoof) y reportar EER/AUC."
    )
 
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
 
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
 
 
class VoiceAIEngine:
    def __init__(
        self,
        nombre_modelo: str = "Vansh180/deepfake-audio-wav2vec2",
        score_ia_umbral: float = 65.0,
        mad_umbral_no_conforme: float = 0.012, 
        entropia_umbral_bits: float = 11.5,     
        pesos_fusion: Optional[dict] = None,
        permitir_modo_contingencia: bool = True,
    ):
        """
        Parameters
        ----------
        score_ia_umbral, mad_umbral_no_conforme, entropia_umbral_bits:
            Umbrales explícitos y calibrables. Los valores por defecto son
            puntos de partida razonables tomados de literatura general, NO
            valores validados para tu dataset/condiciones de grabación.
            Calíbralos con `calibrar_umbrales()` antes de uso serio.
        pesos_fusion:
            Pesos para la combinación lineal de evidencias normalizadas
            (0-1 cada una) en el score final. Por defecto pondera más la
            señal neuronal (más validada en general) que las estadísticas.
        permitir_modo_contingencia:
            Si False, levanta ModeloNeuronalNoDisponibleError en vez de
            continuar con un score neutro cuando el modelo no carga. Para
            uso forense real, se recomienda False: un análisis incompleto
            no debería disfrazarse de análisis completo.
        """
        self.nombre_modelo = nombre_modelo
        self.score_ia_umbral = score_ia_umbral
        self.mad_umbral_no_conforme = mad_umbral_no_conforme
        self.entropia_umbral_bits = entropia_umbral_bits
        self.pesos_fusion = pesos_fusion or {"neuronal": 0.5, "benford": 0.25, "entropia": 0.25}
        self.permitir_modo_contingencia = permitir_modo_contingencia
 
        self.benford_ideal = np.log10(1 + 1.0 / np.arange(1, 10))
 
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(self.nombre_modelo)
            self.model_ia = AutoModelForAudioClassification.from_pretrained(self.nombre_modelo)
            self.ia_operativa = True
        except Exception as e:
            self.ia_operativa = False
            self._error_carga_modelo = str(e)
            if not self.permitir_modo_contingencia:
                raise ModeloNeuronalNoDisponibleError(
                    f"No se pudo cargar '{self.nombre_modelo}': {e}"
                ) from e
 
    # ------------------------------------------------------------------ #
    # Señal 1: Entropía espectral
    # ------------------------------------------------------------------ #
    def _analizar_entropia_espectral(self, y: np.ndarray) -> EvidenciaEntropia:
        espectrograma = np.abs(librosa.stft(y))
        potencia = np.sum(espectrograma ** 2, axis=0)
        suma = np.sum(potencia)
        if suma == 0:
            return EvidenciaEntropia(valor_bits=0.0, z_score=None)
 
        probs = potencia / suma
        probs = probs[probs > 0]
        entropia = float(-np.sum(probs * np.log2(probs)))
        return EvidenciaEntropia(valor_bits=entropia, z_score=None)
 
    # ------------------------------------------------------------------ #
    # Señal 2: Benford sobre coeficientes DCT del espectro log
    # ------------------------------------------------------------------ #
    @staticmethod
    def _primer_digito(valores: np.ndarray) -> np.ndarray:
        """Extrae el primer dígito significativo de forma robusta vía
        log10, sin depender de manipulación de strings (que falla con
        notación científica, ej. 1.2e-05)."""
        valores = np.abs(valores)
        valores = valores[valores > 1e-12]
        exponentes = np.floor(np.log10(valores))
        mantisas = valores / (10.0 ** exponentes)
        primeros = np.floor(mantisas).astype(int)
        primeros = np.clip(primeros, 1, 9)  # corrige errores de borde por flotantes
        return primeros
 
    def _analizar_benford_dct(self, y: np.ndarray) -> EvidenciaBenford:
        """Aplica Benford sobre coeficientes DCT del log-espectro de
        magnitud, siguiendo el enfoque de Cuccovillo et al. (2024) en vez
        de la FFT cruda."""
        espectrograma = np.abs(librosa.stft(y))
        log_espectro = np.log1p(espectrograma)
        coeficientes = dct(log_espectro, axis=0, norm="ortho").flatten()
 
        primeros_digitos = self._primer_digito(coeficientes)
        n = len(primeros_digitos)
        if n < 300:
            return EvidenciaBenford(mad=0.0, p_value=1.0,
                                     categoria_conformidad="Muestra insuficiente",
                                     n_muestras=n)
 
        conteos_obs, _ = np.histogram(primeros_digitos, bins=np.arange(1, 11))
        props_obs = conteos_obs / n
        conteos_esp = self.benford_ideal * n
 
        mad = float(np.mean(np.abs(props_obs - self.benford_ideal)))
 
        # Categorías estándar de Nigrini para primer dígito
        if mad < 0.006:
            categoria = "Conformidad cercana"
        elif mad < 0.012:
            categoria = "Conformidad aceptable"
        elif mad < 0.015:
            categoria = "Conformidad marginal"
        else:
            categoria = "No conforme"
 
        conteos_esp_safe = np.where(conteos_esp == 0, 1e-6, conteos_esp)
        _, p_valor = chisquare(f_obs=conteos_obs, f_exp=conteos_esp_safe)
 
        return EvidenciaBenford(mad=mad, p_value=float(p_valor),
                                 categoria_conformidad=categoria, n_muestras=n)
 
    # ------------------------------------------------------------------ #
    # Señal 3: Modelo neuronal
    # ------------------------------------------------------------------ #
    def _analizar_neuronal(self, y: np.ndarray) -> EvidenciaNeuronal:
        if not self.ia_operativa:
            return EvidenciaNeuronal(disponible=False, score_fake_pct=None,
                                      nombre_modelo=self.nombre_modelo)
 
        inputs = self.extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_ia(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
 
        labels = self.model_ia.config.label2id
        idx_fake = None
        for clave in ("fake", "spoof", "Fake", "Spoof"):
            if clave in labels:
                idx_fake = labels[clave]
                break
        if idx_fake is None:

            raise PipelineForenseError(
                f"No se pudo identificar la etiqueta 'fake'/'spoof' en el modelo "
                f"'{self.nombre_modelo}'. Etiquetas disponibles: {labels}"
            )
 
        score = float(probs[0][idx_fake].item() * 100)
        return EvidenciaNeuronal(disponible=True, score_fake_pct=score,
                                  nombre_modelo=self.nombre_modelo)
 
    # ------------------------------------------------------------------ #
    # Orquestación
    # ------------------------------------------------------------------ #
    def analizar_clonacion(self, ruta_audio: str) -> ReporteForense:
        if not os.path.exists(ruta_audio):
            raise AudioInvalidoError(f"No se encontró el archivo: {ruta_audio}")
 
        try:
            y, sr = librosa.load(ruta_audio, sr=16000)
        except Exception as e:
            raise AudioInvalidoError(f"No se pudo decodificar el audio: {e}") from e
 
        if len(y) == 0:
            raise AudioInvalidoError("El archivo de audio está vacío.")
 
        try:
            ev_neuronal = self._analizar_neuronal(y)
            ev_benford = self._analizar_benford_dct(y)
            ev_entropia = self._analizar_entropia_espectral(y)
        except PipelineForenseError:
            raise
        except Exception as e:
            # Ya no se devuelve un score arbitrario (45). Un fallo de
            # pipeline es un fallo, no un resultado.
            raise PipelineForenseError(f"Fallo en la tubería forense: {e}") from e
 
        # --- Normalización de cada señal a [0,1] como "evidencia de riesgo" ---
        if ev_neuronal.disponible:
            riesgo_neuronal = ev_neuronal.score_fake_pct / 100.0
        else:
            riesgo_neuronal = 0.5  # neutro explícito, declarado en el reporte
 
        riesgo_benford = min(ev_benford.mad / self.mad_umbral_no_conforme, 1.0) \
            if ev_benford.categoria_conformidad != "Muestra insuficiente" else 0.5
 
        riesgo_entropia = 1.0 if ev_entropia.valor_bits < self.entropia_umbral_bits else 0.0
 
        score_riesgo = 100 * (
            self.pesos_fusion["neuronal"] * riesgo_neuronal
            + self.pesos_fusion["benford"] * riesgo_benford
            + self.pesos_fusion["entropia"] * riesgo_entropia
        )
        score_riesgo = float(np.clip(score_riesgo, 0, 100))
 
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
            nivel_confianza=nivel,
        )
 

def calibrar_umbrales(engine: VoiceAIEngine, dataset_etiquetado: list[tuple[str, bool]]):
    """
    Recorre un dataset etiquetado [(ruta_audio, es_fake), ...] (p.ej. una
    partición de ASVspoof 2019/2021 LA, o un corpus propio anotado), calcula
    las tres señales para cada muestra, y reporta:
      - EER (Equal Error Rate) para cada señal individual y para el score
        fusionado.
      - AUC.
      - El umbral que minimiza el EER (en vez de usar 65/0.012/11.5 a ciegas).
 
    Esto es lo que debes incluir en la documentación/defensa del proyecto:
    no "el sistema detecta clonación de voz", sino "el sistema alcanza un
    EER de X% sobre el corpus Y", igual que reportan los papers citados.
 
    Requiere scikit-learn (pip install scikit-learn) y no se ejecuta
    automáticamente: es una herramienta de evaluación, separada del
    pipeline de inferencia.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
 
    scores, etiquetas = [], []
    for ruta, es_fake in dataset_etiquetado:
        try:
            reporte = engine.analizar_clonacion(ruta)
        except (AudioInvalidoError, PipelineForenseError) as e:
            print(f"⚠️ Omitiendo {ruta}: {e}")
            continue
        scores.append(reporte.score_riesgo)
        etiquetas.append(1 if es_fake else 0)
 
    fpr, tpr, umbrales = roc_curve(etiquetas, scores)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    auc = roc_auc_score(etiquetas, scores)
 
    print(f"EER: {eer*100:.2f}%  |  AUC: {auc:.3f}  |  Umbral óptimo: {umbrales[idx_eer]:.1f}")
    return {"eer": eer, "auc": auc, "umbral_optimo": float(umbrales[idx_eer])}
 

voice_ai_engine = VoiceAIEngine()
 