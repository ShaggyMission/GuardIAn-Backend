# CallShield Forense

CallShield Forense es una API en FastAPI para analisis forense de audios orientada a detectar posibles intentos de clonacion de voz, transcribir el contenido hablado y evaluar tecnicas de ingenieria social en conversaciones.

El proyecto expone un flujo de analisis que combina varios motores:

1. `voice_engine`: detecta si el audio parece real o sintetico usando un modelo neuronal SLS.
2. `whisper_engine`: transcribe el audio a texto usando Whisper Small.
3. `social_engine`: revisa el texto transcrito en busca de patrones de fraude e ingenieria social.
4. `utils.py`: consolida el riesgo final y genera recomendaciones.

## Objetivo

El objetivo del sistema es entregar un resultado rapido y centralizado para audios sospechosos, por ejemplo:

- audios de WhatsApp
- llamadas grabadas
- mensajes de voz reenviados
- evidencia de posibles estafas telefonicas

La API no sustituye una pericia forense certificada. El resultado debe entenderse como un analisis de screening o primera revision tecnica.

## Estructura general

```text
app/
  main.py
  config.py
  security.py
  utils.py
  database.py
  models.py
  motores/
    whisper_engine.py
    social_engine.py
    voice_engine/
      engine.py
      config_produccion.json
      sls_best.pth
```

## Flujo del sistema

1. El usuario envia un archivo de audio a la API.
2. La API valida el archivo con `security.py`.
3. `voice_engine` procesa el audio y calcula la probabilidad de clonacion.
4. `whisper_engine` transcribe el contenido.
5. `social_engine` analiza la transcripcion para detectar tacticas de manipulacion.
6. `utils.py` combina las evidencias y genera el nivel de riesgo final.

## Requisitos

El proyecto usa principalmente:

- Python 3.11 o superior
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Librosa
- SoundFile
- Scikit-learn
- SQLAlchemy

Las dependencias estan listadas en `requirements.txt`.

## Instalacion

1. Crear y activar un entorno virtual.
2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Ejecucion local

Levanta la API con:

```powershell
uvicorn app.main:app --reload
```

Luego abre:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## Endpoints principales

### `GET /`

Devuelve un estado general del backend y lista los motores cargados.

### `POST /api/v1/analisis/forense`

Endpoint principal del sistema forense completo.

Entrada:

- archivo de audio en form-data con la clave `file`

Salida:

- transcripcion
- metricas del motor neuronal
- analisis social
- riesgo global
- recomendaciones

### `GET /api/v1/voice-engine/health`

Devuelve el estado del modulo `voice_engine`, el checkpoint usado y la configuracion de produccion.

## Ejemplo de respuesta general

```json
{
  "archivo_procesado": "audio.wav",
  "transcripcion_whisper": "por favor haz la transferencia ahora mismo",
  "metricas": {
    "motor1_voz_ia": 91.3,
    "nivel_confianza_voz": "Audio compatible con clonacion",
    "motor3_ingenieria_social": 74,
    "riesgo_global": 79,
    "nivel": "ALTO / MUY SOSPECHOSO"
  },
  "desglose_tacticas": {
    "urgencia": 80,
    "solicitud_economica": 90
  },
  "recomendaciones_seguridad": [
    "Existe una alta probabilidad de intento de fraude.",
    "No realice transferencias ni envie dinero sin verificar la identidad por otro medio."
  ]
}
```

## Modulos del proyecto

### `app/main.py`

Define la aplicacion FastAPI, registra CORS y expone el endpoint forense principal.

### `app/config.py`

Centraliza la configuracion general de entorno, rutas y directorios de trabajo.

### `app/security.py`

Valida los archivos de audio permitidos.

### `app/utils.py`

Fusiona el score del motor de voz con el analisis social para calcular el riesgo final.

### `app/motores/whisper_engine.py`

Transcribe audios usando Whisper.

### `app/motores/social_engine.py`

Detecta indicios de fraude o ingenieria social en el texto.

### `app/motores/voice_engine/`

Contiene la version de produccion del modelo neuronal de voz, su checkpoint y su API propia.

## Consideraciones tecnicas

- El sistema espera audios que puedan ser leidos por `librosa`.
- El motor de voz usa el archivo `sls_best.pth` ubicado dentro de `app/motores/voice_engine`.
- La configuracion de produccion se controla con `config_produccion.json`.
- Si el modelo no puede cargarse, la API responde con error para evitar devolver resultados falsamente validos.

## Probar rapidamente

El analisis completo se hace con:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/analisis/forense" -F "file=@C:\ruta\audio.wav"
```

## Notas de uso

- Mantener `sls_best.pth` junto a `config_produccion.json`.
- Verificar que las dependencias de audio esten instaladas correctamente.
- Para despliegue real, se recomienda correr el servicio con un servidor ASGI estable y no solo con `--reload`.

## DB (SQLite + SQLAlchemy)

Para dotar al sistema de validez pericial y permitir un histórico local sin fricciones, se implementó una **Arquitectura de Persistencia No Invasiva** basada en una **Tabla Líquida**.

### Características de Diseño
- **Anonimato Forense:** No se almacenan datos personales (nombres o correos). La identidad y pertenencia de los datos se ligan directamente a un identificador único de hardware (`uuid_dispositivo`) enviado por el frontend multimedia.
- **Estructura Líquida:** Las evidencias se guardan serializadas en texto plano mediante `json.dumps()` dentro de la columna `resultado_json`. Esto blinda la base de datos contra cambios o actualizaciones futuras en las métricas de los motores de IA.
- **Tolerancia a Fallos:** El guardado local se ejecuta en caliente de forma asíncrona/pasiva al final de la tubería multimedia. Si la base de datos llegara a bloquearse, el sistema captura el error en los logs pero jamás interrumpe la respuesta HTTP del usuario.

### Endpoints Forenses Añadidos
- `POST /api/v1/analisis/forense`: Recibe el archivo de audio junto al parámetro `uuid_dispositivo` en el cuerpo del formulario para procesar y almacenar el registro.
- `GET /api/v1/analisis/historial/{uuid_dispositivo}`: Recupera cronológicamente el histórico de reportes de un dispositivo específico, desempaquetando el JSON nativo mediante `json.loads()`.
- `GET /api/v1/analisis/auditoria`: Endpoint global plano diseñado para peritos judiciales, permitiendo extraer la cadena de custodia completa del sistema.