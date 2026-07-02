import os
import sys
from transformers import pipeline

# 💡 TRUCO SOBERANO: Forzamos la ruta de FFmpeg en las variables de entorno de Python
# Asegúrate de que tu carpeta descomprimida esté exactamente en C:\ffmpeg
ruta_ffmpeg = r"C:\Users\edgar.garzon\Videos\ffmpeg\ffmpeg-8.1.2-essentials_build\bin"
if os.path.exists(ruta_ffmpeg) and ruta_ffmpeg not in os.environ["PATH"]:
    os.environ["PATH"] += os.path.pathsep + ruta_ffmpeg

class WhisperEngine:
    def __init__(self):
        print("⏳ [IA] Inicializando Motor 2: Transcriptor Whisper (openai/whisper-tiny)...")
        # Usamos whisper-tiny por velocidad y bajo consumo de recursos en desarrollo
        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-tiny"
        )
        print("✅ [IA] Motor 2 (Whisper) cargado exitosamente.")

    def transcribir(self, ruta_audio: str) -> str:
            """
            Recibe la ruta absoluta de un archivo de audio y devuelve su transcripción en texto.
            """
            if not os.path.exists(ruta_audio):
                raise FileNotFoundError(f"No se encontró el archivo de audio en: {ruta_audio}")
            
            try:
                print(f"🎙️ [Whisper] Procesando señales de audio en: {os.path.basename(ruta_audio)}")
                
                # 💡 SOLUCIÓN: Añadimos return_timestamps=True para permitir audios largos (> 30s)
                resultado = self.pipe(ruta_audio, return_timestamps=True)
                
                texto = resultado.get("text", "").strip()
                return texto
            except Exception as e:
                print(f"❌ [Whisper] Error durante la transcripción: {str(e)}")
                raise e

# Instancia única (Singleton) para no recargar el modelo en cada petición HTTP
whisper_engine = WhisperEngine()