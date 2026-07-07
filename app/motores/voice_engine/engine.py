from __future__ import annotations

import dataclasses
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2Config, Wav2Vec2Model


DEFAULT_CONFIG = {
    "umbral_decision": 0.999979,
    "modelo_checkpoint": "sls_best.pth",
    "sample_rate_entrada": 16000,
    "max_len_muestras": 64600,
}


@dataclasses.dataclass
class EvidenciaNeuronal:
    disponible: bool
    score_fake_pct: Optional[float]
    nombre_modelo: str


@dataclasses.dataclass
class ReporteForense:
    archivo: str
    evidencia_neuronal: EvidenciaNeuronal
    score_riesgo: float
    nivel_confianza: str
    advertencia: str
    umbral_decision: float
    prob_real: float
    prob_fake: float
    prediccion: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class SLSModel(nn.Module):
    def __init__(self, device: Optional[torch.device] = None, backbone_config: Optional[Wav2Vec2Config] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.ssl_model = Wav2Vec2Model(backbone_config or _build_wav2vec2_config())
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def get_atten_f(self, layer_results):
        poollayer, fullf = [], []
        for layer in layer_results:
            pooled = layer.transpose(1, 2)
            pooled = F.adaptive_avg_pool1d(pooled, 1)
            pooled = pooled.transpose(1, 2)
            poollayer.append(pooled)
            fullf.append(layer.unsqueeze(1))
        return torch.cat(poollayer, dim=1), torch.cat(fullf, dim=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        outputs = self.ssl_model(x, output_hidden_states=True)
        layer_results = outputs.hidden_states[1:]
        y0, fullfeature = self.get_atten_f(layer_results)
        y0 = self.sig(self.fc0(y0))
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = torch.sum(fullfeature * y0, 1).unsqueeze(1)
        out = self.selu(self.first_bn(fullfeature))
        out = F.max_pool2d(out, (3, 3))
        out = torch.flatten(out, 1)
        out = self.selu(self.fc1(out))
        out = self.selu(self.fc3(out))
        return self.logsoftmax(out)


def _build_wav2vec2_config() -> Wav2Vec2Config:
    try:
        return AutoConfig.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            local_files_only=True,
        )
    except Exception:
        return Wav2Vec2Config(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            feat_proj_dropout=0.0,
            final_dropout=0.1,
            layerdrop=0.05,
            layer_norm_eps=1e-5,
            feat_extract_norm="group",
            feat_extract_activation="gelu",
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=False,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            vocab_size=32,
            classifier_proj_size=256,
            do_stable_layer_norm=True,
            use_weighted_layer_sum=False,
        )


def _load_json_config(config_path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            config.update(loaded)
    return config


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            candidate = obj.get(key)
            if isinstance(candidate, dict):
                return candidate
        if all(hasattr(v, "shape") for v in obj.values()):
            return obj
    raise ValueError("El checkpoint no contiene un state_dict reconocido.")


class VoiceAIEngine:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.config_path = self.base_dir / "config_produccion.json"
        self.checkpoint_path = self.base_dir / "sls_best.pth"
        self.config = _load_json_config(self.config_path)
        self.umbral_decision = float(self.config["umbral_decision"])
        self.sample_rate_entrada = int(self.config["sample_rate_entrada"])
        self.max_len_muestras = int(self.config["max_len_muestras"])
        self.model_name = str(self.config.get("model_name", "XLS-R-SLS"))
        self.advertencia = (
            "Este resultado es un score de riesgo de screening. "
            "No constituye prueba pericial certificada."
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._model: Optional[SLSModel] = None
        self._load_error: Optional[str] = None

    @property
    def listo(self) -> bool:
        return self._model is not None and self._load_error is None

    @property
    def error_carga(self) -> Optional[str]:
        return self._load_error

    def _crear_modelo(self) -> SLSModel:
        modelo = SLSModel(device=self.device)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No se encontro el checkpoint en {self.checkpoint_path}")

        raw_checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(raw_checkpoint)
        state_dict = _strip_module_prefix(state_dict)

        missing, unexpected = modelo.load_state_dict(state_dict, strict=False)
        if missing:
            detalle = {
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            }
            raise RuntimeError(
                f"No se pudo cargar el checkpoint con compatibilidad suficiente: {detalle}"
            )

        modelo = modelo.to(self.device)
        modelo.eval()
        return modelo

    def _asegurar_modelo(self) -> SLSModel:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                self._model = self._crear_modelo()
                self._load_error = None
            except Exception as exc:
                self._model = None
                self._load_error = str(exc)
                raise

        return self._model

    def _preparar_audio(self, ruta_audio: str) -> torch.Tensor:
        y, _ = librosa.load(ruta_audio, sr=self.sample_rate_entrada)
        if len(y) == 0:
            raise ValueError("El audio esta vacio.")

        if len(y) >= self.max_len_muestras:
            y = y[: self.max_len_muestras]
        else:
            repeticiones = int(self.max_len_muestras / len(y)) + 1
            y = np.tile(y, repeticiones)[: self.max_len_muestras]

        audio = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        return audio.unsqueeze(0).to(self.device)

    def _inferir(self, ruta_audio: str) -> ReporteForense:
        modelo = self._asegurar_modelo()
        entrada = self._preparar_audio(ruta_audio)

        with torch.no_grad():
            logits = modelo(entrada)
            probs = torch.exp(logits)[0]

        prob_fake = float(probs[0].item())
        prob_real = float(probs[1].item())
        prediccion = "real" if prob_real >= self.umbral_decision else "fake"
        score_riesgo = round(prob_fake * 100.0, 1)
        nivel_confianza = (
            "Audio compatible con voz real"
            if prediccion == "real"
            else "Audio compatible con clonacion"
        )

        return ReporteForense(
            archivo=os.path.basename(ruta_audio),
            evidencia_neuronal=EvidenciaNeuronal(
                disponible=True,
                score_fake_pct=round(prob_fake * 100.0, 1),
                nombre_modelo=self.model_name,
            ),
            score_riesgo=score_riesgo,
            nivel_confianza=nivel_confianza,
            advertencia=self.advertencia,
            umbral_decision=self.umbral_decision,
            prob_real=round(prob_real, 6),
            prob_fake=round(prob_fake, 6),
            prediccion=prediccion,
        )

    def analizar_clonacion(self, ruta_audio: str) -> ReporteForense:
        if not os.path.exists(ruta_audio):
            raise FileNotFoundError(f"No se encontro el archivo: {ruta_audio}")

        try:
            return self._inferir(ruta_audio)
        except Exception as exc:
            raise RuntimeError(f"Fallo en el analisis del audio: {exc}") from exc

    def estado(self) -> dict[str, Any]:
        return {
            "listo": self.listo,
            "checkpoint": str(self.checkpoint_path),
            "configuracion": {
                "umbral_decision": self.umbral_decision,
                "sample_rate_entrada": self.sample_rate_entrada,
                "max_len_muestras": self.max_len_muestras,
            },
            "error_carga": self._load_error,
        }


voice_ai_engine = VoiceAIEngine()

