import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = os.getenv("APP_NAME", "CallShield Forense")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Umbrales de riesgo
    UMBRAL_RIESGO_CRITICO: int = int(os.getenv("UMBRAL_RIESGO_CRITICO", 75))
    
    # Rutas absolutas base
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    TEMP_DIR: str = os.path.join(BASE_DIR, "app", "temp")

    class Config:
        case_sensitive = True

settings = Settings()

# Asegurar que las carpetas de almacenamiento existan al arrancar la app
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)