# voice_engine

Este modulo implementa el motor neuronal de produccion para analisis de clonacion de voz.

Su funcion es recibir un archivo de audio, normalizarlo, cargar el modelo entrenado `sls_best.pth` y devolver una prediccion basada en la salida del clasificador.

## Que contiene este modulo

```text
app/motores/voice_engine/
  api.py
  engine.py
  __init__.py
  config_produccion.json
  sls_best.pth
```

## Archivos principales

### `engine.py`

Contiene la logica principal del motor:

- carga la configuracion de produccion
- reconstruye la arquitectura `SLSModel`
- carga el checkpoint `sls_best.pth`
- prepara el audio
- ejecuta inferencia
- devuelve un reporte estructurado

### `api.py`

Expone el motor como API FastAPI:

- `GET /api/v1/voice-engine/health`
- `POST /api/v1/voice-engine/analizar`

### `config_produccion.json`

Archivo de configuracion del motor. Define:

- `umbral_decision`
- `modelo_checkpoint`
- `sample_rate_entrada`
- `max_len_muestras`

### `sls_best.pth`

Checkpoint entrenado del modelo de voz.

## Como funciona

1. Se recibe un archivo de audio por HTTP.
2. Se valida que el archivo tenga una extension de audio permitida.
3. El audio se carga con `librosa` y se remuestrea a `16000 Hz`.
4. El vector se ajusta a `64600` muestras.
5. El modelo `SLSModel` genera logits para dos clases:
   - `real`
   - `fake`
6. Se calcula:
   - `prob_real`
   - `prob_fake`
   - `prediccion`
   - `score_riesgo`

## Modelo usado

La arquitectura reproduce la estructura usada en entrenamiento:

- backbone `Wav2Vec2Model`
- bloques de atencion sobre capas ocultas
- capas densas finales
- salida con `LogSoftmax`

El motor espera un checkpoint compatible con esta arquitectura.

## Configuracion de produccion

Ejemplo:

```json
{
  "umbral_decision": 0.999979,
  "modelo_checkpoint": "sls_best.pth",
  "sample_rate_entrada": 16000,
  "max_len_muestras": 64600
}
```

### Significado de cada campo

- `umbral_decision`: umbral para decidir si el audio se clasifica como `real`.
- `modelo_checkpoint`: nombre del archivo `.pth` a cargar.
- `sample_rate_entrada`: frecuencia de muestreo esperada.
- `max_len_muestras`: longitud objetivo del audio procesado.

## Respuesta del endpoint

La API retorna un JSON con:

- `archivo`
- `evidencia_neuronal`
- `score_riesgo`
- `nivel_confianza`
- `advertencia`
- `umbral_decision`
- `prob_real`
- `prob_fake`
- `prediccion`

## Ejemplo de uso

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/voice-engine/analizar" -F "file=@C:\ruta\audio.wav"
```

## Endpoint de salud

```http
GET /api/v1/voice-engine/health
```

Devuelve informacion util para diagnostico:

- si el motor esta listo
- ruta del checkpoint
- parametros cargados
- error de carga, si existe

## Dependencias implicadas

Este modulo depende de:

- `torch`
- `transformers`
- `librosa`
- `numpy`
- `fastapi`

## Recomendaciones

- No mover `sls_best.pth` fuera de esta carpeta sin actualizar `config_produccion.json`.
- Mantener el mismo esquema de arquitectura al reentrenar o reemplazar el checkpoint.
- Si cambias el umbral, valida nuevamente el comportamiento del motor con audios reales y falsos.

## Problemas comunes

### El modelo no carga

Revisa:

- que exista `sls_best.pth`
- que `config_produccion.json` tenga el nombre correcto del checkpoint
- que el entorno tenga `torch` y `transformers`

### El endpoint devuelve error 503

Significa que el motor no pudo inicializarse correctamente y se esta protegiendo la API de entregar resultados incompletos.
