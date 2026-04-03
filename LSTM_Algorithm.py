import numpy as np
import unicodedata
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Data_Manage import Data_Manage


class LSTM_Algorithm:

    def __init__(
        self,
        archivo_csv,
        target,
        sequence_length=8,
        train_ratio=0.8,
        data_mode="tabular",
        n_restarts=1,
    ):
        self.archivo = archivo_csv
        self.target = target
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.data_mode = data_mode
        self.n_restarts = n_restarts
        self.model = None
        self.datos_preprocesados = None
        self.gestor_datos = None

    def _construir_modelo(self, input_shape):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.LSTM(
                    units=96,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    unit_forget_bias=True,
                    dropout=0.25,
                    recurrent_dropout=0.15,
                    kernel_regularizer=tf.keras.regularizers.l2(8e-5),
                    recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                    return_sequences=True,
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.LSTM(
                    units=48,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    unit_forget_bias=True,
                    dropout=0.20,
                    recurrent_dropout=0.10,
                    kernel_regularizer=tf.keras.regularizers.l2(8e-5),
                    recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                    return_sequences=False,
                ),
                tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(8e-5)),
                tf.keras.layers.Dropout(0.20),
                tf.keras.layers.Dense(1),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mae"])
        return model

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
            f"No se encontro el archivo {self.archivo}"
        )

    def _preparar_datos(self):
        archivo_resuelto = self._resolver_archivo()
        gestor = Data_Manage(
            archivo_resuelto,
            self.target,
            sequence_length=self.sequence_length,
            coverage_threshold_secuencial=0.6,
            resample_freq_secuencial=None,
            coverage_threshold_tabular=0.7,
            split_estrategia="aleatorio",
        )

        if self.data_mode == "secuencial":
            X_train, X_test, y_train, y_test = gestor.preparar_datos_secuenciales(
                sequence_length=self.sequence_length,
                train_ratio=self.train_ratio,
                escalar=True,
            )
        else:
            X_train_2d, X_test_2d, y_train, y_test = gestor.preparar_datos_supervisado(
                train_ratio=self.train_ratio,
                escalar=True,
            )
            # LSTM sobre un paso temporal con features enriquecidas suele generalizar mejor aquí.
            X_train = X_train_2d.reshape(X_train_2d.shape[0], 1, X_train_2d.shape[1])
            X_test = X_test_2d.reshape(X_test_2d.shape[0], 1, X_test_2d.shape[1])

        datos = (X_train, X_test, y_train, y_test)
        self.gestor_datos = gestor
        self.datos_preprocesados = datos
        return datos
        

    def _entrenar(
        self, X_train, y_train, epochs=80, batch_size=32, validation_split=0.2
    ):
        n_valid = max(64, int(len(X_train) * validation_split))
        if len(X_train) - n_valid < 64:
            n_valid = max(16, int(len(X_train) * 0.1))

        X_fit = X_train[:-n_valid]
        y_fit = y_train[:-n_valid]
        X_valid = X_train[-n_valid:]
        y_valid = y_train[-n_valid:]

        semillas = [42 + i * 17 for i in range(max(1, self.n_restarts))]
        mejor_modelo = None
        mejor_val_loss = float("inf")

        for idx, semilla in enumerate(semillas, start=1):
            tf.keras.utils.set_random_seed(semilla)
            modelo = self._construir_modelo((X_train.shape[1], X_train.shape[2]))

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=8,
                min_delta=1e-4,
                restore_best_weights=True,
            )
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )

            historial = modelo.fit(
                X_fit,
                y_fit,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_valid, y_valid),
                shuffle=False,
                callbacks=[early_stopping, lr_scheduler],
                verbose=1,
            )

            val_min = min(historial.history.get("val_loss", [float("inf")]))
            if val_min < mejor_val_loss:
                mejor_val_loss = val_min
                mejor_modelo = modelo

            print(f"[LSTM restart {idx}/{len(semillas)}] best val_loss={val_min:.6f}")

        self.model = mejor_modelo

    def _predecir(self, X):
        return self.model.predict(X).flatten()

    def ejecutar(self, epochs=60, batch_size=32) -> dict:
        X_train, X_test, y_train, y_test = self._preparar_datos()

        self._entrenar(X_train, y_train, epochs=epochs, batch_size=batch_size)

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
        }
