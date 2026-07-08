import os
from fastapi import HTTPException, UploadFile

AUDIOS_PERMITIDOS = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg"
}

def validar_archivo_audio(file: UploadFile) -> str:
    """
    Valida la extensión y tipo de contenido del archivo cargado.
    Devuelve la extensión detectada si es válido, de lo contrario lanza una excepción HTTP 400.
    """
    content_type = file.content_type
    filename = file.filename.lower()
    
    if content_type not in AUDIOS_PERMITIDOS:
        extension_valida = any(filename.endswith(ext) for ext in AUDIOS_PERMITIDOS.values())
        if not extension_valida:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no soportado ({content_type}). Solo se permiten archivos de audio (.mp3, .wav, .m4a)"
            )
            
    _, ext = os.path.splitext(filename)
    return ext if ext else ".mp3"