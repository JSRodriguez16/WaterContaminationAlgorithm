"""
SVM_Algorithm.py
----------------
Implementación de Máquina de Vectores de Soporte (SVR) para la predicción
de la Demanda Química de Oxígeno (DQO) en estaciones de monitoreo de Colombia.

El modelo utiliza Support Vector Regression (SVR) con kernel RBF, combinado con
búsqueda de hiperparámetros mediante RandomizedSearchCV para optimizar C, epsilon
y gamma. Sigue exactamente las mismas convenciones de preprocesamiento, escalado,
partición y reporte de métricas que los demás modelos del proyecto
(LinearRegression_Algorithm, XGBoost_Algorithm, LSTM_Algorithm).

Referencia metodológica:
    Koli et al. (2025). "A Hybrid Approach to Water Quality Classification
    Using SVM and Xgboost Method." IJRSI, vol. XII, n.º VI, pp. 1083-1086.
    doi: 10.51244/IJRSI.2025.12060080
"""

import unicodedata
from pathlib import Path

import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from Data_Manage import Data_Manage
from Uncertainty_Analysis import calcular_metricas_incertidumbre


class SVM_Algorithm:
    """
    Modelo SVR con kernel RBF para predicción de DQO.

    Parámetros
    ----------
    archivo_csv : str
        Ruta al CSV histórico de calidad de agua.
    target : str
        Nombre de la variable objetivo (p. ej. "DEMANDA QUIMICA DE OXIGENO").
    train_ratio : float
        Proporción de datos destinada a entrenamiento (default 0.8).
    buscar_hiperparametros : bool
        Si True ejecuta RandomizedSearchCV; si False usa los valores por defecto
        ajustados manualmente. Útil para reproducibilidad rápida.
    n_iter_busqueda : int
        Número de combinaciones evaluadas por RandomizedSearchCV.
    cv_folds : int
        Número de pliegues de validación cruzada durante la búsqueda.
    random_state : int
        Semilla para reproducibilidad.
    """

    # Hiperparámetros por defecto encontrados empíricamente para datos de DQO.
    _DEFAULTS = {
        "C": 10.0,
        "epsilon": 0.1,
        "gamma": "scale",
        "kernel": "rbf",
        "cache_size": 500,
        "max_iter": 5000,
    }

    def __init__(
        self,
        archivo_csv: str,
        target: str,
        train_ratio: float = 0.8,
        buscar_hiperparametros: bool = True,
        n_iter_busqueda: int = 30,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.archivo = archivo_csv
        self.target = target
        self.train_ratio = train_ratio
        self.buscar_hiperparametros = buscar_hiperparametros
        self.n_iter_busqueda = n_iter_busqueda
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.model: SVR | None = None
        self.preprocesador: Data_Manage | None = None
        self.feature_cols: list[str] = []

        # Información de diagnóstico guardada tras el entrenamiento.
        self._mejores_hiperparametros: dict = {}
        self._mejor_score_cv: float | None = None
        self._ruta_csv_resuelta: str | None = None

    # ------------------------------------------------------------------
    # Utilidades de resolución de rutas (mismo patrón que XGBoost/Lineal)
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

        raise FileNotFoundError(
            f"No se encontró el archivo CSV '{self.archivo}'. "
            "Verifica nombre/ruta (por ejemplo, acentos como historica vs histórica)."
        )

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------

    def _preparar_datos(self) -> tuple:
        """
        Carga y preprocesa los datos usando Data_Manage en modo tabular,
        con la misma configuración que XGBoost (split aleatorio + log-transform).
        """
        archivo_resuelto = self._resolver_archivo()
        self._ruta_csv_resuelta = archivo_resuelto

        pre = Data_Manage(
            archivo_resuelto,
            self.target,
            split_estrategia="aleatorio",
        )
        self.preprocesador = pre

        X_train, X_test, y_train, y_test = pre.preparar_datos_supervisado(
            train_ratio=self.train_ratio,
            escalar=True,
        )
        _, feature_cols, _ = pre._preparar_dataset_modelo(modo="tabular")
        self.feature_cols = feature_cols

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Construcción y entrenamiento
    # ------------------------------------------------------------------

    def _construir_modelo_base(self) -> SVR:
        """Construye SVR con hiperparámetros por defecto."""
        return SVR(**self._DEFAULTS)

    def _buscar_hiperparametros(self, X_train: np.ndarray, y_train: np.ndarray) -> SVR:
        """
        Realiza búsqueda aleatoria de hiperparámetros sobre un subconjunto
        de entrenamiento para mantener tiempos razonables.

        SVR tiene complejidad O(n²) a O(n³) en el número de muestras, por lo
        que se limita la búsqueda a un subconjunto representativo cuando el
        dataset es grande (>3000 filas).
        """
        MAX_MUESTRAS_BUSQUEDA = 3000
        rng = np.random.default_rng(self.random_state)

        if len(X_train) > MAX_MUESTRAS_BUSQUEDA:
            idx = rng.choice(len(X_train), size=MAX_MUESTRAS_BUSQUEDA, replace=False)
            X_bus = X_train[idx]
            y_bus = y_train[idx]
        else:
            X_bus = X_train
            y_bus = y_train

        espacio = {
            "C":       loguniform(1e-1, 1e3),   # [0.1, 100]
            "epsilon": loguniform(1e-3, 1.0),   # [0.001, 1]
            "gamma":   ["scale", "auto"] + list(loguniform(1e-4, 1.0).rvs(8, random_state=self.random_state)),
        }

        svr_base = SVR(
            kernel="rbf",
            cache_size=500,
            max_iter=5000,
        )

        busqueda = RandomizedSearchCV(
            estimator=svr_base,
            param_distributions=espacio,
            n_iter=self.n_iter_busqueda,
            scoring="neg_root_mean_squared_error",
            cv=self.cv_folds,
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        busqueda.fit(X_bus, y_bus)

        self._mejores_hiperparametros = busqueda.best_params_
        self._mejor_score_cv = float(-busqueda.best_score_)  # RMSE positivo

        return busqueda.best_estimator_

    def _entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entrena el modelo SVR con o sin búsqueda de hiperparámetros."""
        if self.buscar_hiperparametros:
            self.model = self._buscar_hiperparametros(X_train, y_train)
            # Re-ajusta sobre el conjunto completo de entrenamiento con los
            # mejores hiperparámetros encontrados en el subconjunto de búsqueda.
            self.model.fit(X_train, y_train)
        else:
            self.model = self._construir_modelo_base()
            self.model.fit(X_train, y_train)
            self._mejores_hiperparametros = self._DEFAULTS.copy()

    def _predecir(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llame a ejecutar() primero."
            )
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Importancia de variables mediante permutación
    # ------------------------------------------------------------------

    def _calcular_importancia_permutacion(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_features: int = 15,
        n_repeticiones: int = 5,
    ) -> list[dict]:
        """
        Calcula la importancia de variables mediante permutación (model-agnostic).

        A diferencia de SHAP/TreeExplainer, este método es compatible con
        cualquier estimador, incluido SVR. Mide cuánto aumenta el MAE al
        permutar aleatoriamente cada variable, manteniendo el resto fijo.

        Parámetros
        ----------
        X_test : np.ndarray
            Conjunto de prueba escalado.
        y_test : np.ndarray
            Valores reales escalados del objetivo.
        max_features : int
            Número máximo de variables a reportar (las más importantes).
        n_repeticiones : int
            Número de permutaciones por variable (promediado para estabilidad).

        Retorna
        -------
        list[dict] con {"feature", "importance"} ordenado descendentemente.
        """
        if self.model is None or not self.feature_cols:
            return []

        rng = np.random.default_rng(self.random_state)
        mae_base = float(mean_absolute_error(y_test, self._predecir(X_test)))

        importancias = []
        for i in range(X_test.shape[1]):
            deltas = []
            for _ in range(n_repeticiones):
                X_perm = X_test.copy()
                X_perm[:, i] = rng.permutation(X_perm[:, i])
                mae_perm = float(mean_absolute_error(y_test, self._predecir(X_perm)))
                deltas.append(mae_perm - mae_base)
            importancias.append(float(np.mean(deltas)))

        importancias = np.array(importancias)
        orden = np.argsort(importancias)[::-1]
        top = orden[: min(max_features, len(orden))]

        return [
            {
                "feature": self.feature_cols[idx],
                "importance": float(importancias[idx]),
            }
            for idx in top
            if importancias[idx] > 0  # Solo variables con impacto positivo
        ]

    # ------------------------------------------------------------------
    # Ejecución principal
    # ------------------------------------------------------------------

    def ejecutar(self) -> dict:
        """
        Ejecuta el pipeline completo: preprocesamiento → entrenamiento →
        evaluación → análisis de incertidumbre → importancia de variables.

        Retorna
        -------
        dict con las mismas claves que XGBoost_Algorithm y
        LinearRegression_Algorithm para facilitar la comparación en
        Diagnosis_Algorithms.py.
        """
        X_train, X_test, y_train, y_test = self._preparar_datos()

        self._entrenar(X_train, y_train)

        y_pred_train_scaled = self._predecir(X_train)
        y_pred_scaled = self._predecir(X_test)

        if self.preprocesador is not None:
            y_pred_train = self.preprocesador.desescalar_target(y_pred_train_scaled)
            y_train_real = self.preprocesador.desescalar_target(y_train)
            y_pred      = self.preprocesador.desescalar_target(y_pred_scaled)
            y_test_real = self.preprocesador.desescalar_target(y_test)
        else:
            y_pred_train = y_pred_train_scaled
            y_train_real = y_train
            y_pred       = y_pred_scaled
            y_test_real  = y_test

        mae_train  = mean_absolute_error(y_train_real, y_pred_train)
        rmse_train = float(np.sqrt(mean_squared_error(y_train_real, y_pred_train)))
        mae        = mean_absolute_error(y_test_real, y_pred)
        rmse       = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
        r2         = r2_score(y_test_real, y_pred)

        incertidumbre = calcular_metricas_incertidumbre(
            y_train_real=y_train_real,
            y_pred_train=y_pred_train,
            y_test_real=y_test_real,
            y_pred_test=y_pred,
        )

        importancia = self._calcular_importancia_permutacion(X_test, y_test)

        return {
            # Tamaños de partición
            "n_train": len(X_train),
            "n_test":  len(X_test),
            # Métricas de entrenamiento
            "mae_train":  mae_train,
            "rmse_train": rmse_train,
            # Métricas de prueba
            "y_test": y_test_real,
            "y_pred": y_pred,
            "mae":    mae,
            "rmse":   rmse,
            "r2":     r2,
            # Diagnóstico del modelo SVM
            "svm_kernel":              "rbf",
            "svm_mejores_hiperparametros": self._mejores_hiperparametros,
            "svm_mejor_rmse_cv":       self._mejor_score_cv,
            "svm_n_support_vectors":   int(self.model.support_vectors_.shape[0]) if self.model else None,
            # Importancia de variables por permutación
            "svm_importancia_variables": importancia,
            # Análisis de incertidumbre (igual que demás modelos)
            **incertidumbre,
        }
