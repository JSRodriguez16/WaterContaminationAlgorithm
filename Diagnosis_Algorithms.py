from LSTM_Algorithm import LSTM_Algorithm


def mostrar_resultados(resultados: dict) -> None:
    print("Resultados del Modelo LSTM")
    print("-" * 35)
    print(f"  Muestras de entrenamiento: {resultados['n_train']}")
    print(f"  Muestras de prueba: {resultados['n_test']}")
    print("-" * 35)
    print("Precision del Modelo")
    print("-" * 50)
    print(f"MAE (Error absoluto medio): {resultados['mae']:.4f}")
    print(f"RMSE (Raiz del error cuadratico): {resultados['rmse']:.4f}")
    print(f"R Cuadrado (Coeficiente de determinacion): {resultados['r2']:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    lstm = LSTM_Algorithm(
        "Data_historica_de_calidad_de_agua_20260223.csv",
        "DEMANDA QUIMICA DE OXIGENO"
    )
    resultados = lstm.ejecutar()
    mostrar_resultados(resultados)
