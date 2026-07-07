# 🛡️ CallShield Forense - Backend API

Servidor analítico síncrono de alto rendimiento desarrollado en **FastAPI (Python)**. Forma parte del ecosistema de ciberseguridad multipropósito de **CallShield**, encargado del procesamiento digital de señales de audio, transcripción semántica de alta fidelidad mediante IA local y evaluación pericial de amenazas de extorsión telefónica o clonación de voz.

---

## 📂 Arquitectura del Módulo Whisper Forense

El subsistema de análisis de texto y transcripción opera de manera modular bajo un flujo secuencial síncrono independiente del usuario. A diferencia de transcriptores planos tradicionales, este cuadrante ejecuta auditorías temporales avanzadas:

1. **Detección de Idioma Agnóstico:** El decodificador analiza los primeros segundos de la señal, infiere dinámicamente el idioma original del hablante y activa el diccionario adecuado mediante probabilidades de pesos de la red neuronal.
2. **Evaluación de Coherencia por Haz:** Incorpora decodificación de parámetros mediante `num_beams` acoplado al modelo avanzado **`openai/whisper-small` (244M parámetros)**, resolviendo de manera matemática distorsiones fonéticas y sinalefas acústicas naturales de la voz humana.
3. **Métrica Cruzada de Inactividad (VAD vs Cabecera):** Contrasta de forma nativa los metadatos binarios del archivo (`wave`) contra los bloques legibles calculados por la IA. Permite determinar la **Tasa de Silencios Neta de la llamada**, indicador pericial clave para perfilar automatizaciones delictivas o bots de extorsión.

---

## 🛠️ Requisitos e Instalación del Sistema

### 1. Descarga y Ubicación Obligatoria de FFmpeg

El motor de inteligencia artificial requiere de binarios externos para descomprimir matrices multimedia. Para asegurar la portabilidad del proyecto entre las computadoras de todo el equipo, se ha estandarizado una ruta absoluta independiente del usuario del sistema operativo:

1. Descarga el archivo oficial comprimido: [ffmpeg-release-essentials.zip](https://www.gyan.dev/ffmpeg/builds/).
2. Extrae el contenido directamente en la raíz de tu **Disco Local C:**.
3. Asegúrate de renombrar la carpeta para que la ruta hacia los ejecutables sea exactamente la siguiente:

```text
C:\ffmpeg\bin
```

*(Nota: El código cuenta con una lógica preventiva que buscará automáticamente esta ruta estándar tanto en entornos de oficina como residenciales).*

### 2. Clonación y Configuración del Entorno Virtual

Abre tu terminal de Windows PowerShell dentro del directorio principal del backend (`CallShield-Forense`) y ejecuta la siguiente secuencia de comandos:

```
# 1. Crear la carpeta del entorno virtual local
python -m venv venv

# 2. Desbloquear temporalmente las políticas de ejecución restrictivas de Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 3. Activar el entorno virtual (verás el indicador '(venv)' al inicio de la línea)
.\venv\Scripts\activate

# 4. Actualizar el gestor de paquetes e instalar las librerías neuronales

pip install --upgrade pip
o
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 3. Configuración de Credenciales Locales

Copia la plantilla de ejemplo y crea tu archivo de variables de entorno definitivo:

```
copy .env.example .env
```

---

## 🚀 Despliegue y Pruebas en Tiempo Real

Para encender el servidor y activar el recargado automático ante cualquier cambio de desarrollo, ejecuta:

```
uvicorn app.main:app --reload
o
python app/pruebas-whisper/test_whisper.py
```

Una vez que observes en consola el mensaje `Application startup complete`, abre tu navegador web preferido e ingresa al siguiente enlace para interactuar de forma gráfica con la API:

👉 **http://127.0.0.1:8000/docs**

### 🧪 Cómo realizar un Test Forense en el Swagger UI

1. Despliega el bloque interactivo correspondiente al endpoint **`POST /api/v1/analisis/forense`**.
2. Presiona el botón **"Try it out"** en la esquina superior derecha.
3. En el campo `file`, presiona *Seleccionar archivo* y carga un audio de prueba en formato `.wav` o `.mp3`.
4. Haz clic en el botón azul gigante **"Execute"**.
5. Revisa el *Response body* devuelto: el sistema te entregará un JSON dinámico estructurado con los puntajes de clonación acústica, tácticas semánticas de ingeniería social y el nuevo desglose forense avanzado de métricas de silencio generados por el cuadrante de Whisper.

---
