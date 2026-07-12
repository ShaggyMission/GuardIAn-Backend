import os
import json
from openai import OpenAI
from app.config import settings

class OpenAIForensicEngine:
    def __init__(self):
        print("[IA] Inicializando Motores 2 y 3: OpenAI Cloud Services...")
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            print("⚠️ [ADVERTENCIA] OPENAI_API_KEY no detectada en el entorno.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            print("[IA] Conexión a OpenAI establecida correctamente.")

    def transcribir_audio(self, ruta_audio: str) -> str:
        """Motor 2: Transcripción rápida mediante Whisper API."""
        if not self.client:
            return "Error: API de OpenAI no configurada."
        
        try:
            print(f"[OpenAI Whisper] Transcribiendo archivo: {os.path.basename(ruta_audio)}")
            with open(ruta_audio, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="es"
                )
            return transcription.text.strip()
        except Exception as e:
            print(f"[OpenAI Whisper] Error en transcripción: {str(e)}")
            return ""

    def analizar_texto_social(self, texto: str) -> dict:
        """Motor 3: Análisis semántico forense estructurado con GPT-4o-mini."""
        if not self.client or not texto:
            return self._retorno_vacio()

        prompt_sistema = """
        Eres un perito forense analizando una transcripción de audio (llamada o WhatsApp) para detectar estafas, extorsión o ingeniería social.
        Analiza el texto y devuelve ÚNICAMENTE un objeto JSON válido con la siguiente estructura exacta, sin texto adicional ni bloques markdown:
        {
            "fraude_detectado": booleano (true si hay intento claro de manipulación/estafa, false si es normal),
            "confianza_fraude": entero del 0 al 100,
            "riesgo_social": entero del 0 al 100,
            "nivel_riesgo": string ("BAJO", "MEDIO", "ALTO" o "CRÍTICO"),
            "tacticas": lista de strings con las tácticas detectadas (ej. "urgencia", "solicitud_economica", "suplantacion", "amenaza", "manipulacion_emocional", "aislamiento", "obtencion_informacion"),
            "es_narracion_tts": booleano (true si el texto describe explícitamente acciones de lectura automatizada, asistentes de voz, instrucciones de dictado o herramientas de conversión de texto a voz como 'estaré encantado de leerlo en voz alta', false de lo contrario)
        }
        """

        try:
            print("[OpenAI GPT] Ejecutando análisis semántico de ingeniería social...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Texto a analizar: '{texto}'"}
                ],
                temperature=0.2,
                response_format={ "type": "json_object" }
            )
            
            contenido = response.choices[0].message.content
            resultado = json.loads(contenido)
            return resultado

        except Exception as e:
            print(f"[OpenAI GPT] Error en análisis semántico: {str(e)}")
            return self._retorno_vacio()

    def _retorno_vacio(self):
        return {
            "fraude_detectado": False,
            "confianza_fraude": 0,
            "riesgo_social": 0,
            "nivel_riesgo": "BAJO",
            "tacticas": [],
            "es_narracion_tts": False
        }

openai_engine = OpenAIForensicEngine()