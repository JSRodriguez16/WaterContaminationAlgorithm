import numpy as np
import unicodedata
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Asegúrate de importar la clase correcta
from Data_Manage import Data_Manage

class XGBoost_Algorithm:

    def __init__(self, archivo_csv, target):
        self.archivo = archivo_csv
        self.target = target
        self.preprocesador = None
        
        self.model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.2,
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
        )

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        texto_norm = unicodedata.normalize("NFD", texto)
        texto_sin_acentos = "".join(c for c in texto_norm if unicodedata.category(c) != "Mn")
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
            raise FileNotFoundError(f"No existe el directorio para buscar el CSV: {directorio_busqueda}")

        nombre_objetivo = self._normalizar_texto(ruta.name)
        for archivo in directorio_busqueda.iterdir():
            if archivo.is_file() and self._normalizar_texto(archivo.name) == nombre_objetivo:
                return str(archivo)

        raise FileNotFoundError(
            f"No se encontró el archivo CSV '{self.archivo}'. "
            "Verifica nombre/ruta (por ejemplo, acentos como historica vs histórica)."
        )

    def _preparar_datos(self):
        archivo_resuelto = self._resolver_archivo()
        
        pre = Data_Manage(
            archivo_resuelto,
            self.target,
            split_estrategia="aleatorio",
        )
        self.preprocesador = pre

        # Para boosting, la vista tabular con lags y rolling suele rendir mejor que aplanar secuencias.
        return pre.preparar_datos_supervisado(train_ratio=0.8, escalar=True)

    def _entrenar(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def _predecir(self, X):
        return self.model.predict(X)

    def ejecutar(self) -> dict:
        X_train, X_test, y_train, y_test = self._preparar_datos()

        self._entrenar(X_train, y_train)

        y_pred_scaled = self._predecir(X_test)

        if self.preprocesador is not None:
            y_pred = self.preprocesador.desescalar_target(y_pred_scaled)
            y_test_real = self.preprocesador.desescalar_target(y_test)
        else:
            y_pred = y_pred_scaled
            y_test_real = y_test

        mae  = mean_absolute_error(y_test_real, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
        r2   = r2_score(y_test_real, y_pred)

        return {
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "y_test":     y_test_real,
            "y_pred":     y_pred,
            "mae":        mae,
            "rmse":       rmse,
            "r2":         r2,
        }