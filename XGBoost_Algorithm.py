import numpy as np
import unicodedata
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:  # pragma: no cover - dependencia opcional en tiempo de ejecución
    shap = None

# Asegúrate de importar la clase correcta
from Data_Manage import Data_Manage
from Uncertainty_Analysis import calcular_metricas_incertidumbre

class XGBoost_Algorithm:

    def __init__(self, archivo_csv, target):
        self.archivo = archivo_csv
        self.target = target
        self.preprocesador = None
        
        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=6,
            gamma=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=60,
            tree_method='hist',
            n_jobs=-1,
        )
        self._valid_ratio = 0.15
        self._train_eval_tail = []
        self._valid_eval_tail = []
        self._best_iteration = None
        self._train_mejora_val_empeora = False
        self._ruta_csv_resuelta = None
        self._mostrar_shap_interactivo = True

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
        self._ruta_csv_resuelta = archivo_resuelto
        
        pre = Data_Manage(
            archivo_resuelto,
            self.target,
            split_estrategia="aleatorio",
        )
        self.preprocesador = pre

        # Para boosting, la vista tabular con lags y rolling suele rendir mejor que aplanar secuencias.
        X_train, X_test, y_train, y_test = pre.preparar_datos_supervisado(train_ratio=0.8, escalar=True)
        _, feature_cols, _ = pre._preparar_dataset_modelo(modo="tabular")
        return X_train, X_test, y_train, y_test, feature_cols

    def _entrenar(self, X_train, y_train):
        n_total = len(X_train)
        n_valid = max(40, int(n_total * self._valid_ratio))

        if n_total - n_valid < 100:
            # Fallback si hay pocos datos para reservar validación.
            self.model.fit(X_train, y_train)
            return

        X_fit = X_train[:-n_valid]
        y_fit = y_train[:-n_valid]
        X_valid = X_train[-n_valid:]
        y_valid = y_train[-n_valid:]

        self.model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_fit, y_fit), (X_valid, y_valid)],
            verbose=False,
        )

        resultados = self.model.evals_result()
        rmse_train = resultados.get("validation_0", {}).get("rmse", [])
        rmse_valid = resultados.get("validation_1", {}).get("rmse", [])

        if rmse_train and rmse_valid:
            self._best_iteration = int(np.argmin(rmse_valid))
            self._train_eval_tail = rmse_train[-10:]
            self._valid_eval_tail = rmse_valid[-10:]

            # Señal de overfitting solo si la mejora/empeoramiento es relevante (>1%).
            mejora_train_rel = (rmse_train[self._best_iteration] - rmse_train[-1]) / max(rmse_train[self._best_iteration], 1e-12)
            empeora_valid_rel = (rmse_valid[-1] - rmse_valid[self._best_iteration]) / max(rmse_valid[self._best_iteration], 1e-12)
            self._train_mejora_val_empeora = (
                mejora_train_rel > 0.01
                and empeora_valid_rel > 0.01
            )

    def _predecir(self, X):
        return self.model.predict(X)

    def _generar_importancia_shap(self, X_train, feature_cols, max_muestras=200, max_features=15):
        if shap is None:
            raise RuntimeError(
                "La dependencia 'shap' no está instalada. Instálala para generar el gráfico de importancia."
            )

        if X_train is None or len(X_train) == 0 or not feature_cols:
            return {"plot_path": None, "feature_importance": []}

        n_muestras = min(max_muestras, len(X_train))
        rng = np.random.default_rng(42)
        if len(X_train) > n_muestras:
            indices = rng.choice(len(X_train), size=n_muestras, replace=False)
            X_muestra = X_train[indices]
        else:
            X_muestra = X_train

        explainer = None
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_muestra)
        except Exception:
            X_perm = X_muestra[: min(80, len(X_muestra))]
            background = X_perm[: min(40, len(X_perm))]
            explainer = shap.Explainer(self.model.predict, background)
            max_evals = max(2 * X_perm.shape[1] + 1, 600)
            explanation = explainer(X_perm, max_evals=max_evals)  # type: ignore[arg-type]
            shap_values = explanation.values

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]

        importancia = np.abs(shap_values).mean(axis=0)
        orden = np.argsort(importancia)[::-1]

        top_indices = orden[: min(max_features, len(orden))]
        top_features = [
            {"feature": feature_cols[idx], "importance": float(importancia[idx])}
            for idx in top_indices
        ]

        plot_path = None
        plot_mode = "interactivo"

        figuras_altura = max(4.5, 0.38 * len(top_indices) + 1.5)
        fig, ax = plt.subplots(figsize=(10.5, figuras_altura))
        valores_plot = importancia[top_indices][::-1]
        etiquetas_plot = [feature_cols[idx] for idx in top_indices][::-1]
        ax.barh(etiquetas_plot, valores_plot, color="#1f77b4")
        ax.set_title("Importancia global de variables basada en SHAP")
        ax.set_xlabel("Media de |valor SHAP|")
        ax.set_ylabel("Variable")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        fig.tight_layout()

        if self._mostrar_shap_interactivo:
            try:
                plt.show()
            except Exception:
                salida_base = Path(self._ruta_csv_resuelta).resolve().parent if self._ruta_csv_resuelta else Path(__file__).resolve().parent
                directorio_salida = salida_base / "shap_outputs"
                directorio_salida.mkdir(parents=True, exist_ok=True)
                plot_path = directorio_salida / "xgboost_shap_importance.png"
                fig.savefig(plot_path, dpi=220, bbox_inches="tight")
                plot_mode = "archivo"
            finally:
                plt.close(fig)
        else:
            salida_base = Path(self._ruta_csv_resuelta).resolve().parent if self._ruta_csv_resuelta else Path(__file__).resolve().parent
            directorio_salida = salida_base / "shap_outputs"
            directorio_salida.mkdir(parents=True, exist_ok=True)
            plot_path = directorio_salida / "xgboost_shap_importance.png"
            fig.savefig(plot_path, dpi=220, bbox_inches="tight")
            plot_mode = "archivo"
            plt.close(fig)

        return {
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_mode": plot_mode,
            "feature_importance": top_features,
        }

    def ejecutar(self) -> dict:
        X_train, X_test, y_train, y_test, feature_cols = self._preparar_datos()

        self._entrenar(X_train, y_train)

        y_pred_train_scaled = self._predecir(X_train)
        y_pred_scaled = self._predecir(X_test)

        if self.preprocesador is not None:
            y_pred_train = self.preprocesador.desescalar_target(y_pred_train_scaled)
            y_train_real = self.preprocesador.desescalar_target(y_train)
            y_pred = self.preprocesador.desescalar_target(y_pred_scaled)
            y_test_real = self.preprocesador.desescalar_target(y_test)
        else:
            y_pred_train = y_pred_train_scaled
            y_train_real = y_train
            y_pred = y_pred_scaled
            y_test_real = y_test

        mae_train = mean_absolute_error(y_train_real, y_pred_train)
        rmse_train = float(np.sqrt(mean_squared_error(y_train_real, y_pred_train)))
        mae  = mean_absolute_error(y_test_real, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
        r2   = r2_score(y_test_real, y_pred)

        incertidumbre = calcular_metricas_incertidumbre(
            y_train_real=y_train_real,
            y_pred_train=y_pred_train,
            y_test_real=y_test_real,
            y_pred_test=y_pred,
        )
        shap_resultados = self._generar_importancia_shap(X_train, feature_cols)

        return {
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "mae_train":  mae_train,
            "rmse_train": rmse_train,
            "xgb_best_iteration": self._best_iteration,
            "xgb_train_mejora_val_empeora": self._train_mejora_val_empeora,
            "xgb_train_rmse_tail": self._train_eval_tail,
            "xgb_valid_rmse_tail": self._valid_eval_tail,
            "y_test":     y_test_real,
            "y_pred":     y_pred,
            "mae":        mae,
            "rmse":       rmse,
            "r2":         r2,
            "shap_plot_path": shap_resultados["plot_path"],
            "shap_plot_mode": shap_resultados["plot_mode"],
            "shap_feature_importance": shap_resultados["feature_importance"],
            **incertidumbre,
        }
