import os
from transformers import pipeline

class SocialEngine:
    def __init__(self):
        print("⏳ [IA] Inicializando Motor 3: Analizador de Ingeniería Social (BART-Large-MNLI)...")
        try:
            # Inicialización del pipeline zero-shot para clasificación de texto
            self.clasificador = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
            print("✅ [IA] Motor 3 (Ingeniería Social) cargado exitosamente.")
        except Exception as e:
            print(f"❌ [SocialEngine Error] No se pudo cargar el modelo BART: {str(e)}")
            self.clasificador = None

    def analizar_texto(self, texto: str) -> dict:
        """
        Analiza semánticamente la transcripción buscando patrones de extorsión o fraude.
        """
        if not self.clasificador or not texto.strip():
            return {"amenaza": 0, "suplantacion": 0, "urgencia": 0}

        # Etiquetas forenses para identificar las tácticas delictivas
        etiquetas = [
            "amenaza o violencia", 
            "suplantación de identidad (familiar/autoridad)", 
            "solicitud de dinero o depósito urgente",
            "falsa emergencia o secuestro"
        ]

        try:
            resultado = self.clasificador(texto, etiquetas, multi_label=True)
            desglose = {}
            for etiqueta, score in zip(resultado["labels"], resultado["scores"]):
                if "amenaza" in etiqueta:
                    desglose["amenaza"] = int(score * 100)
                elif "suplantación" in etiqueta:
                    desglose["suplantacion"] = int(score * 100)
                elif "dinero" in etiqueta:
                    desglose["urgencia"] = int(score * 100)
                elif "emergencia" in etiqueta:
                    desglose["falsa_emergencia"] = int(score * 100)
            
            return desglose
        except Exception as e:
            print(f"❌ [SocialEngine] Error en la inferencia NLP: {str(e)}")
            return {"amenaza": 0, "suplantacion": 0, "urgencia": 0}

social_engine = SocialEngine()