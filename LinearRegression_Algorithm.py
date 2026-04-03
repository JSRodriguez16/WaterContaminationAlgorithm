import unicodedata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data_Manage import Data_Manage


class LinearRegression_Algorithm:
    def __init__(self, archivo_csv: str, target: str, train_ratio: float = 0.8):
        self.archivo = archivo_csv
        self.target = target
        self.train_ratio = train_ratio

        self.model: LassoCV | None = None
        self.datos_preprocesados: tuple | None = None
        self.gestor_datos: Data_Manage | None = None
        self.feature_cols: list[str] = []


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

    def _preparar_datos(self) -> tuple:
        
        archivo_resuelto = self._resolver_archivo()
        gestor = Data_Manage(
            archivo_resuelto,
            self.target,
            split_estrategia="aleatorio",
            transformar_target_log=False,
        )

        datos = gestor.preparar_datos_supervisado(
            train_ratio=self.train_ratio,
            escalar=True,
        )

        df_model, feature_cols, _ = gestor._preparar_dataset_modelo()
        self.feature_cols = feature_cols

        self.gestor_datos = gestor
        self.datos_preprocesados = datos
        return datos

    def _construir_modelo(self) -> LassoCV:
        return LassoCV(
            alphas=np.logspace(-5, 1, 20),
            max_iter=40000,
            random_state=42,
        )

    def _entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.model is None:
            self.model = self._construir_modelo()
        self.model.fit(X_train, y_train)

    def _predecir(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def obtener_coeficientes(self) -> dict:
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
            "alpha": round(float(self.model.alpha_), 8),
            "coeficientes": coefs,
        }

    def ejecutar(self) -> dict:
        X_train, X_test, y_train, y_test = self._preparar_datos()

        self._entrenar(X_train, y_train)

        y_pred_train_scaled = self._predecir(X_train)
        y_pred_scaled = self._predecir(X_test)

        if self.gestor_datos is not None:
            y_pred_train = self.gestor_datos.desescalar_target(y_pred_train_scaled)
            y_train_real = self.gestor_datos.desescalar_target(y_train)
            y_pred = self.gestor_datos.desescalar_target(y_pred_scaled)
            y_test_real = self.gestor_datos.desescalar_target(y_test)
        else:
            y_pred_train = y_pred_train_scaled
            y_train_real = y_train
            y_pred = y_pred_scaled
            y_test_real = y_test

        mae_train = mean_absolute_error(y_train_real, y_pred_train)
        rmse_train = float(np.sqrt(mean_squared_error(y_train_real, y_pred_train)))
        mae = mean_absolute_error(y_test_real, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
        r2 = r2_score(y_test_real, y_pred)

        return {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mae_train": mae_train,
            "rmse_train": rmse_train,
            "y_test": y_test_real,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "coeficientes": self.obtener_coeficientes(),
        }