from transformers import pipeline


# =========================
# MODELO
# =========================

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


# =========================
# ETAPA 1
# DETECCIÓN GENERAL
# =========================

MANIPULATION_LABELS = [
    "fraude o intento de manipulación",
    "conversación normal"
]


# =========================
# ETAPA 2
# IDENTIFICACIÓN DE TÁCTICAS
# =========================

SOCIAL_LABELS = [

    "urgencia o presión para actuar inmediatamente",
    "solicitud de dinero o transferencia",
    "suplantación de identidad",
    "amenaza o intimidación",
    "manipulación emocional",
    "chantaje emocional",
    "autoridad o institución",
    "culpa",
    "aislamiento o secreto",
    "falsa emergencia",
    "obtención de información confidencial"

]


LABEL_MAPPING = {

    "urgencia o presión para actuar inmediatamente":
        "urgencia",

    "solicitud de dinero o transferencia":
        "solicitud_economica",

    "suplantación de identidad":
        "suplantacion",

    "amenaza o intimidación":
        "amenaza",

    "manipulación emocional":
        "manipulacion_emocional",

    "chantaje emocional":
        "chantaje_emocional",

    "autoridad o institución":
        "autoridad",

    "culpa":
        "culpa",

    "aislamiento o secreto":
        "aislamiento",

    "falsa emergencia":
        "falsa_emergencia",

    "obtención de información confidencial":
        "info_confidencial"

}



def empty_result():

    return {

        "urgencia": 0,
        "solicitud_economica": 0,
        "suplantacion": 0,
        "amenaza": 0,
        "manipulacion_emocional": 0,
        "chantaje_emocional": 0,
        "autoridad": 0,
        "culpa": 0,
        "aislamiento": 0,
        "falsa_emergencia": 0,
        "info_confidencial": 0

    }



# =========================
# MOTOR SOCIAL ENGINE
# =========================

class SocialEngine:


    def __init__(self):

        print(
            "⏳ [IA] Inicializando Motor 3: Ingeniería Social (mDeBERTa)..."
        )


        try:

            self.clasificador = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME
            )


            print(
                "✅ [IA] Motor 3 cargado correctamente."
            )


        except Exception as e:

            print(
                f"❌ Error cargando SocialEngine: {e}"
            )

            self.clasificador = None



    # =====================================================
    # FASE 1
    # DETECTAR SI EXISTE POSIBLE MANIPULACIÓN
    # =====================================================

    def detectar_manipulacion(self, texto):

        if not self.clasificador:
            return {
            "fraude_detectado": False,
            "confianza_fraude": 0
        }


        resultado = self.clasificador(

        texto,

        MANIPULATION_LABELS,

        multi_label=False

    )


        etiqueta = resultado["labels"][0]

        confianza = resultado["scores"][0]


        fraude = (
            etiqueta == "fraude o intento de manipulación"
            and confianza >= 0.60
    )


        return {

        "fraude_detectado": fraude,

        "confianza_fraude": int(confianza * 100)

    }



    # =====================================================
    # FASE 2
    # ANALIZAR TÁCTICAS ESPECÍFICAS
    # =====================================================

    def analizar_tacticas(self, texto):


        resultado = self.clasificador(

            texto,

            SOCIAL_LABELS,

            multi_label=True

        )


        salida = empty_result()



        for label, score in zip(

            resultado["labels"],

            resultado["scores"]

        ):


            key = LABEL_MAPPING.get(label)



            if key:

                valor = int(score * 100)


                # Eliminamos ruido bajo
                if valor >= 50:

                    salida[key] = valor



        return salida




    # =====================================================
    # MÉTODO PRINCIPAL
    # =====================================================

    def analizar_texto(self, texto):


        if not texto or not texto.strip():

            return {

                "fraude_detectado": False,

                "confianza_fraude": 0,

                "tacticas": empty_result()

            }



        primera_etapa = self.detectar_manipulacion(texto)



        if not primera_etapa["fraude_detectado"]:


            return {

                **primera_etapa,

                "tacticas": empty_result()

            }



        tacticas = self.analizar_tacticas(texto)



        return {

            **primera_etapa,

            "tacticas": tacticas

        }



# =========================
# INSTANCIA GLOBAL
# =========================

social_engine = SocialEngine()