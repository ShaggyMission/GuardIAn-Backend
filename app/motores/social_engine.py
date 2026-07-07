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
            "[IA] Inicializando Motor 3: Ingenieria Social (mDeBERTa)..."
        )

        try:

            self.clasificador = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME
            )

            print(
                "[IA] Motor 3 cargado correctamente."
            )


        except Exception as e:

            print(
                f"[SocialEngine] Error cargando SocialEngine: {e}"
            )

            self.clasificador = None



    # =====================================================
    # FASE 1
    # DETECCIÓN DE MANIPULACIÓN
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

            and

            confianza >= 0.70

        )


        return {

            "fraude_detectado": fraude,

             "confianza_fraude": (
                int(confianza * 100)
                if fraude
                else 0
    )

}



    # =====================================================
    # FASE 2
    # ANÁLISIS DE TÁCTICAS
    # =====================================================

    def analizar_tacticas(self, texto):


        resultado = self.clasificador(

            texto,

            SOCIAL_LABELS,

            multi_label=True

        )


        salida = empty_result()



        tacticas_criticas = [

            "solicitud_economica",
            "suplantacion",
            "falsa_emergencia",
            "amenaza",
            "aislamiento",
            "info_confidencial"

        ]



        tacticas_secundarias = [

            "urgencia",
            "manipulacion_emocional",
            "chantaje_emocional",
            "autoridad",
            "culpa"

        ]



        for label, score in zip(

            resultado["labels"],

            resultado["scores"]

        ):


            key = LABEL_MAPPING.get(label)


            if not key:

                continue



            valor = int(score * 100)



            if key in tacticas_criticas:

                if valor >= 45:

                    salida[key] = valor



            elif key in tacticas_secundarias:

                if valor >= 70:

                    salida[key] = valor



        return salida



    # =====================================================
    # FILTRO CONTEXTUAL
    # REDUCE FALSOS POSITIVOS
    # =====================================================

    def filtrar_tacticas_contextuales(self, texto, tacticas):


        texto_lower = texto.lower()



        filtros = {


            "autoridad": [

                "banco",
                "policía",
                "policia",
                "juez",
                "empresa",
                "institución",
                "institucion"

            ],


            "solicitud_economica": [

                "dinero",
                "pago",
                "transferencia",
                "transferir",
                "depósito",
                "deposito"

            ],


            "suplantacion": [

                "soy tu hijo",
                "soy tu hija",
                "soy yo",
                "soy tu mamá",
                "soy tu papá",
                "soy tu hermano",
                "soy tu hermana",
                "soy del banco",
                "soy policía",
                "soy policia",
                "soy familiar",
                "cambié de número",
                "este es mi nuevo número"

],


            "aislamiento": [

                "no le digas a nadie",
                "mantén secreto",
                "no cuentes",
                "solo confía en mí"

            ],


            "info_confidencial": [

                "datos",
                "clave",
                "contraseña",
                "información",
                "informacion"

            ],
            "amenaza": [

                "si no",

                "consecuencias",

                "vas a pagar",

                "te arrepentirás",

                "última advertencia",

                "último aviso",

                "castigo",

                "problemas legales"

            ],
            "culpa": [

                "por tu culpa",

                "es tu culpa",

                "me abandonaste",

                "siempre haces lo mismo",

                "todo es culpa tuya"

            ],
            "chantaje_emocional":[

                "si me quieres",

                "si de verdad me amas",

                "si no haces esto",

                "solo tú puedes ayudarme",

                "depende de ti"

        ],
        "manipulacion_emocional":[

                "estoy desesperado",

                "estoy llorando",

                "me voy a morir",

                "por favor ayúdame",

                "no tengo a nadie",

            "estoy muy mal"

        ],
        "urgencia":[

            "urgente",

            "ahora",

            "ya",

            "de inmediato",

            "en este momento",

            "rápido"

        ],
        }



        resultado = empty_result()



        for tecnica, valor in tacticas.items():


            if valor == 0:

                continue



            if tecnica not in filtros:

                resultado[tecnica] = valor

                continue



            encontrado = any(

                palabra in texto_lower

                for palabra in filtros[tecnica]

            )


            if encontrado:

                resultado[tecnica] = valor



        return resultado
    # =====================================================
    # FASE 3
    # CÁLCULO DE RIESGO SOCIAL
    # =====================================================

    def calcular_riesgo_social(self, confianza_fraude, tacticas):
        """
        Calcula el riesgo final combinando:
        - Confianza del detector general.
        - Severidad de las tácticas detectadas.
        - Combinaciones típicas de ingeniería social.
        """

        if not tacticas:
            return confianza_fraude

        # =====================================
        # PESOS POR TÁCTICA
        # =====================================

        pesos = {

            "solicitud_economica": 1.5,
            "suplantacion": 1.5,
            "falsa_emergencia": 1.4,
            "info_confidencial": 1.4,
            "amenaza": 1.3,
            "aislamiento": 1.3,

            "chantaje_emocional": 1.2,
            "urgencia": 1.1,
            "manipulacion_emocional": 1.1,

            "autoridad": 1.0,
            "culpa": 0.9
        }

        suma = 0
        peso_total = 0

        for tecnica, valor in tacticas.items():

            if valor <= 0:
                continue

            peso = pesos.get(tecnica, 1)

            suma += valor * peso
            peso_total += peso

        if peso_total == 0:
            riesgo_tacticas = 0
        else:
            riesgo_tacticas = suma / peso_total

        # =====================================
        # FACTOR POR CANTIDAD DE TÁCTICAS
        # =====================================

        tecnicas_fuertes = sum(

            1

            for valor in tacticas.values()

            if valor >= 70

        )

        factor = max(0.70, min(tecnicas_fuertes / 5, 1))

        riesgo_tacticas *= factor

        # =====================================
        # RIESGO BASE
        # =====================================

        riesgo_final = (

            confianza_fraude * 0.35

            +

            riesgo_tacticas * 0.65

        )

        # =====================================
        # BONIFICACIONES
        # =====================================

        if (
            tacticas["solicitud_economica"] >= 70 and
            tacticas["urgencia"] >= 70
        ):
            riesgo_final += 5

        if (
            tacticas["suplantacion"] >= 70 and
            tacticas["solicitud_economica"] >= 70
        ):
            riesgo_final += 10

        if (
            tacticas["suplantacion"] >= 70 and
            tacticas["falsa_emergencia"] >= 70
        ):
            riesgo_final += 8

        if (
            tacticas["autoridad"] >= 70 and
            tacticas["info_confidencial"] >= 70
        ):
            riesgo_final += 10

        if (
            tacticas["aislamiento"] >= 70 and
            tacticas["solicitud_economica"] >= 70
        ):
            riesgo_final += 8

        if (
            tacticas["amenaza"] >= 70 and
            tacticas["solicitud_economica"] >= 70
        ):
            riesgo_final += 8

        riesgo_final = max(0, min(riesgo_final, 100))

        return int(riesgo_final)

    # =====================================================
    # NIVEL DE RIESGO
    # =====================================================

    def obtener_nivel_riesgo(self, riesgo):

        if riesgo >= 80:
            return "CRITICO"

        elif riesgo >= 60:
            return "ALTO"

        elif riesgo >= 30:
            return "MEDIO"

        else:
            return "BAJO"

    # =====================================================
    # MÉTODO PRINCIPAL
    # =====================================================

    def analizar_texto(self, texto):

        if not texto or not texto.strip():

            return {

                "fraude_detectado": False,

                "confianza_fraude": 0,

                "riesgo_social": 0,

                "nivel_riesgo": "BAJO",

                "tacticas": empty_result()

            }


        # =====================================
        # ETAPA 1
        # DETECCIÓN GENERAL DE FRAUDE
        # =====================================

        primera_etapa = self.detectar_manipulacion(texto)



        # Si no detecta intento de fraude
        if not primera_etapa["fraude_detectado"]:

            tacticas_modelo = self.analizar_tacticas(texto)

            tacticas = self.filtrar_tacticas_contextuales(
                texto,
                tacticas_modelo
    )

            riesgo_social = self.calcular_riesgo_social(
                 primera_etapa["confianza_fraude"],
                 tacticas
    )

            nivel = self.obtener_nivel_riesgo(
             riesgo_social
    )

        return {

        "fraude_detectado": riesgo_social >= 50,

        "confianza_fraude": primera_etapa["confianza_fraude"],

        "riesgo_social": riesgo_social,

        "nivel_riesgo": nivel,

        "tacticas": tacticas

    }



        # =====================================
        # ETAPA 2
        # DETECCIÓN DE TÁCTICAS
        # =====================================

        tacticas_modelo = self.analizar_tacticas(texto)



        tacticas = self.filtrar_tacticas_contextuales(

            texto,

            tacticas_modelo

        )



        # =====================================
        # ETAPA 3
        # CÁLCULO DE RIESGO SOCIAL
        # =====================================

        riesgo_social = self.calcular_riesgo_social(

            primera_etapa["confianza_fraude"],

            tacticas

        )



        nivel = self.obtener_nivel_riesgo(

            riesgo_social

        )



        return {

            "fraude_detectado": primera_etapa["fraude_detectado"],

            "confianza_fraude": primera_etapa["confianza_fraude"],

            "riesgo_social": riesgo_social,

            "nivel_riesgo": nivel,

            "tacticas": tacticas

        }
    
social_engine = SocialEngine()
