from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from Data_Manage import PreprocesamientoAgua


class LSTM_Algorithm:

    def __init__(self, archivo_csv, target):

        self.archivo = archivo_csv
        self.target = target
        self.model = None

    def _construir_modelo(self, input_shape):

        model = Sequential()

        model.add(
            LSTM(
                64,
                return_sequences=True,
                input_shape=input_shape
            )
        )

        model.add(Dropout(0.2))

        model.add(LSTM(32))

        model.add(Dense(16, activation="relu"))

        model.add(Dense(1))

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        return model

    def preparar_datos(self):

        pre = PreprocesamientoAgua(
            self.archivo,
            self.target
        )

        return pre.procesar()

    def entrenar(
        self,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    ):

        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._construir_modelo(input_shape)

        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

    def evaluar(self, X_test, y_test):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Ejecuta entrenar() primero.")
        return self.model.evaluate(X_test, y_test)

    def predecir(self, X):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Ejecuta entrenar() primero.")
        return self.model.predict(X)

    def ejecutar_pipeline(self):
        X_train, X_test, y_train, y_test = self.preparar_datos()
        self.entrenar(X_train, y_train)
        resultado = self.evaluar(X_test, y_test)

        print("\nEvaluación del modelo:")
        print(resultado)

        return resultado