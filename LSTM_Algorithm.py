import numpy as np
import unicodedata
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from Data_Manage import PreprocesamientoAgua


class LSTM_Algorithm:

    def __init__(self, archivo_csv, target):
        self.archivo = archivo_csv
        self.target = target
        self.model = None
        self.preprocesador = None

    def _construir_modelo(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

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
        pre = PreprocesamientoAgua(archivo_resuelto, self.target)
        self.preprocesador = pre
        return pre.procesar()

    def _entrenar(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.model = self._construir_modelo((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )

    def _predecir(self, X):
        return self.model.predict(X).flatten()

    def ejecutar(self, epochs=30, batch_size=32) -> dict:
        X_train, X_test, y_train, y_test = self._preparar_datos()

        self._entrenar(X_train, y_train, epochs=epochs, batch_size=batch_size)

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