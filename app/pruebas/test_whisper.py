import os
import sys

# 🔄 FIJACIÓN ABSOLUTA DE RUTA: 
# Detectamos dónde está parado este archivo y subimos dos niveles exactos 
# para forzar a Python a encontrar la raíz 'CallShield-Forense'
RUTA_ESTE_ARCHIVO = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.abspath(os.path.join(RUTA_ESTE_ARCHIVO, "..", ".."))

if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

# Ahora Python sí sabe exactamente dónde está la carpeta 'app'
from app.motores.whisper_engine import whisper_engine

# 📂 RUTA EXTERNA ABSOLUTA TOTALMENTE INDEPENDIENTE DE GIT
RUTA_AUDIOS_EXTERNA = r"C:\Users\andyj\Desktop\Proyectos moviles\Audios"
FORMATOS_SOPORTADOS = (".wav", ".mp3", ".m4a", ".caf", ".mp4")

def desplegar_menu_interactivo() -> str:
    """
    Escanea la carpeta externa de la computadora, lista los archivos de audio
    y le pide al usuario que seleccione uno mediante un menú numerado.
    """
    if not os.path.exists(RUTA_AUDIOS_EXTERNA):
        print(f"❌ ERROR CRÍTICO: La carpeta externa no existe en esta PC: {RUTA_AUDIOS_EXTERNA}")
        return ""
    
    # Filtramos los archivos que tengan las extensiones válidas de audio
    audios = [f for f in os.listdir(RUTA_AUDIOS_EXTERNA) if f.lower().endswith(FORMATOS_SOPORTADOS)]
    
    if not audios:
        print(f"⚠️ No se encontraron archivos de audio válidos en: {RUTA_AUDIOS_EXTERNA}")
        return ""
    
    print("\n🎵 ================================================")
    print(f"  MÓDULO DE PRUEBAS INTERACTIVAS - WHISPER FORENSE")
    print("==================================================")
    for indice, archivo in enumerate(audios, start=1):
        print(f" [{indice}] -> {archivo}")
    print("==================================================")
    
    try:
        seleccion = int(input("👉 Escribe el número del archivo que deseas testear: "))
        if 1 <= seleccion <= len(audios):
            return os.path.join(RUTA_AUDIOS_EXTERNA, audios[seleccion - 1])
        else:
            print("❌ Número fuera de rango.")
            return ""
    except ValueError:
        print("❌ Entrada no válida. Por favor introduce un número.")
        return ""

if __name__ == "__main__":
    audio_seleccionado = desplegar_menu_interactivo()
    
    if audio_seleccionado:
        print(f"\n🎙️ [Aislamiento Whisper] Procesando: {os.path.basename(audio_seleccionado)}")
        print("⏳ Ejecutando pipeline neuronal de forma síncrona...")
        
        try:
            # 🚀 LLAMADA EXCLUSIVA A TU APARTADO
            texto_final = whisper_engine.transcribir(audio_seleccionado)
            
            print("\n📝 [TEXTO DETECTADO POR TU MOTOR]:")
            print(f"\"{texto_final}\"")
            
            # Forense: Solicitamos el estado temporal de tu clase
            print("\n📊 [MÉTRICAS DE TU CLASE FORENSE]:")
            metricas = whisper_engine.obtener_metricas_forenses()
            for llave, valor in metricas.items():
                if llave != "segmentacion_lineal":  # Omitimos los chunks largos para ver el resumen limpio
                    print(f" • {llave.replace('_', ' ').capitalize()}: {valor}")
                    
            print("\n✅ Ejecución síncrona en aislamiento completada con éxito.")
            
        except Exception as e:
            print(f"❌ Error en la ejecución del test: {str(e)}")