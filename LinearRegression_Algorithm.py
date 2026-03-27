import unicodedata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data_Manage import Data_Manage


class LinearRegression_Algorithm:
    """
    Modelo de Regresión Lineal Multivariable para predicción de DQO.

    Predice el valor continuo de la Demanda Química de Oxígeno (DQO)
    a partir de parámetros fisicoquímicos del agua.

    El modelo sigue la ecuación:
        DQO = β0 + β1·X1 + β2·X2 + ... + βn·Xn + ε

    Uso:
        modelo = LinearRegression_Algorithm(
            "datos.csv",
            "DEMANDA QUIMICA DE OXIGENO"
        )
        resultados = modelo.ejecutar()
    """

    def __init__(self, archivo_csv: str, target: str, train_ratio: float = 0.8):
        self.archivo = archivo_csv
        self.target = target
        self.train_ratio = train_ratio

        self.model: LinearRegression | None = None
        self.datos_preprocesados: tuple | None = None
        self.gestor_datos: Data_Manage | None = None
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Resolución de ruta del archivo (idéntico a LSTM_Algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        texto_norm = unicodedata.normalize("NFD", texto)
        texto_sin_acentos = "".join(
            c for c in texto_norm if unicodedata.category(c) != "Mn"
        )
        return texto_sin_acentos.casefold()

    def _resolver_archivo(self) -> str:
        ruta = Path(self.archivo)

        if ruta.exists():
            return str(ruta)

        if not ruta.is_absolute():
            base_dir = Path(__file__).resolve().parent
            candidata = base_dir / ruta
            if candidata.exists():
                return str(candidata)
            directorio_busqueda = candidata.parent
        else:
            directorio_busqueda = ruta.parent

        if not directorio_busqueda.exists():
            raise FileNotFoundError(
                f"No existe el directorio para buscar el CSV: {directorio_busqueda}"
            )

        nombre_objetivo = self._normalizar_texto(ruta.name)
        for archivo in directorio_busqueda.iterdir():
            if (
                archivo.is_file()
                and self._normalizar_texto(archivo.name) == nombre_objetivo
            ):
                return str(archivo)

        raise FileNotFoundError(f"No se encontro el archivo {self.archivo}")

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------

    def _preparar_datos(self) -> tuple:
        
        """
        Carga y preprocesa los datos en formato tabular (2D), apropiado
        para regresión lineal. Usa preparar_datos_supervisado() de
        Data_Manage, a diferencia del LSTM que usa datos secuenciales.
        """
        
        archivo_resuelto = self._resolver_archivo()
        gestor = Data_Manage(
            archivo_resuelto,
            self.target,
        )

        # Formato tabular 2D: (n_muestras, n_features) — no requiere secuencias
        datos = gestor.preparar_datos_supervisado(
            train_ratio=self.train_ratio,
            escalar=True,
        )

        # Guardar los nombres de features para los coeficientes
        df_model, feature_cols, _ = gestor._preparar_dataset_modelo()
        self.feature_cols = feature_cols

        self.gestor_datos = gestor
        self.datos_preprocesados = datos
        return datos

    # ------------------------------------------------------------------
    # Construcción y entrenamiento del modelo
    # ------------------------------------------------------------------

    def _construir_modelo(self) -> LinearRegression:
        """
        Instancia el modelo de regresión lineal.
        fit_intercept=True incluye el término independiente β0.
        """
        return LinearRegression(fit_intercept=True)

    def _entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Ajusta los coeficientes β del modelo con los datos de entrenamiento."""
        if self.model is None:
            self.model = self._construir_modelo()
        self.model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def _predecir(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Coeficientes del modelo (interpretabilidad)
    # ------------------------------------------------------------------

    def obtener_coeficientes(self) -> dict:
        """
        Retorna los coeficientes β estimados por variable y el intercepto β0.

        Permite interpretar la contribución individual de cada parámetro
        fisicoquímico sobre el valor predicho de DQO.

        Returns:
            dict con claves:
                'intercepto' (float): β0
                'coeficientes' (dict): {nombre_variable: valor_β}
        """
        if self.model is None:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llame a ejecutar() primero."
            )

        coefs = {
            col: round(float(coef), 6)
            for col, coef in zip(self.feature_cols, self.model.coef_)
        }
        return {
            "intercepto": round(float(self.model.intercept_), 6),
            "coeficientes": coefs,
        }

    # ------------------------------------------------------------------
    # Ejecución principal (interfaz idéntica a LSTM_Algorithm.ejecutar)
    # ------------------------------------------------------------------

    def ejecutar(self) -> dict:
        """
        Ejecuta el pipeline completo:
            1. Carga y preprocesa los datos
            2. Entrena el modelo de regresión lineal
            3. Genera predicciones sobre el conjunto de prueba
            4. Desescala los resultados a la escala original de DQO
            5. Calcula métricas de evaluación

        Returns:
            dict con claves:
                'n_train'  : número de muestras de entrenamiento
                'n_test'   : número de muestras de prueba
                'y_test'   : valores reales de DQO (escala original)
                'y_pred'   : valores predichos de DQO (escala original)
                'mae'      : Error Absoluto Medio
                'rmse'     : Raíz del Error Cuadrático Medio
                'r2'       : Coeficiente de Determinación R²
                'coeficientes': dict con β0 e intercepto por variable
        """
        X_train, X_test, y_train, y_test = self._preparar_datos()

        self._entrenar(X_train, y_train)

        y_pred_scaled = self._predecir(X_test)

        # Desescalar a la unidad original de DQO (mg/L)
        if self.gestor_datos is not None:
            y_pred = self.gestor_datos.desescalar_target(y_pred_scaled)
            y_test_real = self.gestor_datos.desescalar_target(y_test)
        else:
            y_pred = y_pred_scaled
            y_test_real = y_test

        mae = mean_absolute_error(y_test_real, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
        r2 = r2_score(y_test_real, y_pred)

        return {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "y_test": y_test_real,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "coeficientes": self.obtener_coeficientes(),
        }
