# GuardIAn — Backend de Análisis Forense de Audio
 
Este es el motor de backend de **GuardIAn**, encargado de recibir un archivo de audio (llamada, nota de voz, etc.) y determinar, mediante inteligencia artificial y análisis estadístico, **qué tan probable es que se trate de un intento de estafa, extorsión o suplantación de identidad**.
 
El sistema está construido en **FastAPI** y orquesta cuatro motores de análisis independientes que se combinan en un único veredicto de riesgo.
 
---
 
## 🎯 ¿Qué problema resuelve?
 
Las estafas telefónicas modernas combinan dos armas: **voces clonadas con IA** (para suplantar a un familiar, funcionario o entidad bancaria) y **guiones de manipulación psicológica** (urgencia, amenazas, chantaje emocional). GuardIAn ataca ambos frentes a la vez, cruzando evidencia acústica y evidencia lingüística en un solo informe pericial, en lugar de depender de una sola señal aislada.
 
---
 
## 🧩 Arquitectura general
 
El flujo de un audio subido pasa por una tubería (`main.py`) de 4 motores secuenciales:
 
```
Audio subido
   │
   ▼
[Motor 1] voice_engine  →  ¿La voz es clonada/sintética?
   │
   ▼
[Motor 2] whisper_engine →  Transcripción + métricas de habla
   │
   ▼
[Motor 3] social_engine  →  ¿El texto contiene manipulación?
   │
   ▼
[Motor 4] utils (riesgo) →  Fusión final + recomendaciones
   │
   ▼
Veredicto de riesgo global + informe pericial
```
 
---
 
## 🔍 Motor 1 — `voice_engine` (Detección de clonación de voz)
 
Analiza la señal de audio en bruto (no el texto) para detectar si la voz fue generada o clonada artificialmente. No se basa en una sola métrica, sino en **tres evidencias independientes que se fusionan**:
 
1. **Evidencia neuronal**: un modelo de clasificación de audio (`wav2vec2` afinado para detección de *deepfakes*) entrega una probabilidad de que el audio sea falso.
2. **Ley de Benford sobre coeficientes DCT del espectro**: los audios sintéticos tienden a desviarse de la distribución estadística natural que sigue el audio orgánico grabado. Se mide el grado de desviación (MAD) y su significancia estadística (p-valor).
3. **Entropía espectral**: mide la "complejidad" o aleatoriedad del espectro de frecuencias; las voces generadas por IA suelen mostrar patrones espectrales más uniformes/artificiales que una voz humana real.
Estas tres señales se ponderan (por defecto: 50% neuronal, 25% Benford, 25% entropía) en un **score de riesgo de 0 a 100**, junto con un nivel cualitativo de confianza (mínimo, bajo, moderado, alto).
 
> El propio código es honesto sobre sus límites: el resultado se documenta explícitamente como un *score de screening estadístico*, no como prueba pericial certificada, salvo que se calibren los umbrales contra un corpus etiquetado (p. ej. ASVspoof). Incluye además una función `calibrar_umbrales()` para ese propósito.
 
---
 
## 📝 Motor 2 — `whisper_engine` (Transcripción y métricas de habla)
 
Utiliza **Whisper-Small** (OpenAI, vía Hugging Face) para transcribir el audio a texto, y adicionalmente extrae **metadatos forenses de la conversación**:
 
- Duración física real del archivo
- Duración de habla activa vs. tiempo de silencio
- Porcentaje de silencios/pausas en la llamada
- Ritmo del habla (palabras por segundo)
- Segmentación temporal por fragmentos (chunks)
Esta información no solo alimenta la transcripción que analizará el Motor 3, sino que también sirve como evidencia estructural adicional del comportamiento de la llamada (por ejemplo, un ritmo de habla inusualmente rápido o silencios anómalos pueden ser indicios de guiones leídos o de presión psicológica).
 
---
 
## 🧠 Motor 3 — `social_engine` (Ingeniería social y manipulación)
 
Analiza el **texto transcrito** con un modelo de clasificación *zero-shot* (`mDeBERTa-v3` multilingüe) en tres etapas:
 
1. **Detección general**: ¿el texto parece una conversación normal o un intento de fraude/manipulación?
2. **Identificación de tácticas específicas**: urgencia, solicitud de dinero, suplantación de identidad, amenazas, chantaje emocional, invocación de autoridad, culpa, aislamiento/secreto, falsas emergencias, solicitud de datos confidenciales.
3. **Filtro contextual**: cada táctica detectada por el modelo se contrasta contra palabras clave reales del texto, para reducir falsos positivos del clasificador.
Con las tácticas confirmadas, se calcula un **riesgo social ponderado** (no todas las tácticas pesan igual; suplantación y solicitud de dinero pesan más que la culpa, por ejemplo) y se aplican **bonificaciones por combinaciones peligrosas** — por ejemplo, si se detectan *suplantación + solicitud económica* simultáneamente, el riesgo se incrementa adicionalmente, porque juntas son mucho más indicativas de fraude que por separado.
 
---
 
## ⚖️ Motor 4 — Fusión de riesgo y recomendaciones (`utils.py`)
 
Combina el score del Motor 1 (voz sintética) con el riesgo social del Motor 3 en un **riesgo global de 0 a 100**:
 
- Si el riesgo de voz sintética es muy alto (≥70), se le da más peso propio en la fórmula, porque una voz clonada detectada con alta confianza es, por sí sola, una señal muy fuerte de fraude.
- En caso contrario, el riesgo global pondera 25% voz + 75% ingeniería social.
Según el resultado, se asigna un nivel (**Crítico, Alto, Medio, Bajo**) junto con un **protocolo de recomendaciones de seguridad** específico para ese nivel (desde "aislar el archivo y no realizar transferencias" hasta "conservar las pautas estándar de seguridad digital").
 
---
 
## 🛡️ Otras piezas del sistema
 
- **`security.py`**: valida que el archivo subido sea realmente un audio soportado (`.mp3`, `.wav`, `.m4a`, `.ogg`) antes de dejarlo entrar a la tubería de análisis.
- **`config.py`**: centraliza la configuración de la aplicación (nombre, entorno, puertos, umbral de riesgo crítico, rutas de almacenamiento temporal).
- **`main.py`**: expone el endpoint `POST /api/v1/analisis/forense`, que recibe el archivo, lo guarda temporalmente, ejecuta los 4 motores en orden, arma el informe pericial completo (transcripción, métricas, tácticas detectadas, análisis forense de voz y recomendaciones) y **elimina el archivo del servidor al finalizar**, sin dejar rastro persistente del audio subido.
---
 
## 📦 Qué entrega el informe final
 
Por cada audio analizado, el sistema devuelve un informe estructurado que incluye:
 
- La transcripción completa del audio
- El score de voz sintética y su nivel de confianza
- El desglose de tácticas de manipulación detectadas
- El riesgo social y si se detectó fraude
- El riesgo global consolidado y su nivel (Crítico / Alto / Medio / Bajo)
- Las métricas forenses del habla (silencios, ritmo, duración)
- Un conjunto de recomendaciones de seguridad accionables
---
 
## 🧭 Filosofía del proyecto
 
El sistema está diseñado deliberadamente para **no ocultar sus límites**: cuando un modelo no está disponible o un cálculo no puede realizarse con certeza, se declara explícitamente en el reporte en lugar de simular un resultado. La meta es ofrecer una herramienta de **screening y alerta temprana** que ayude a una persona a detectar señales de estafa en segundos, no un veredicto legal definitivo.