from LSTM_Algorithm import LSTM_Algorithm
from LinearRegression_Algorithm import LinearRegression_Algorithm

CSV = "Data_historica_de_calidad_de_agua_20260223.csv"
TARGET = "DEMANDA QUIMICA DE OXIGENO"


# Funciones de visualización de resultados


def mostrar_resultados_lstm(resultados: dict) -> None:
    print("\nResultados del Modelo LSTM")
    print("-" * 35)
    print(f"  Muestras de entrenamiento: {resultados['n_train']}")
    print(f"  Muestras de prueba:        {resultados['n_test']}")
    print("-" * 35)
    print("Precision del Modelo")
    print("-" * 50)
    print(f"MAE  (Error absoluto medio):              {resultados['mae']:.4f}")
    print(f"RMSE (Raiz del error cuadratico medio):   {resultados['rmse']:.4f}")
    print(f"R²   (Coeficiente de determinacion):      {resultados['r2']:.4f}")
    print("-" * 50)


def mostrar_resultados_regresion(resultados: dict) -> None:
    print("\nResultados del Modelo de Regresion Lineal Multivariable")
    print("-" * 55)
    print(f"  Muestras de entrenamiento: {resultados['n_train']}")
    print(f"  Muestras de prueba:        {resultados['n_test']}")
    print("-" * 55)
    print("Precision del Modelo")
    print("-" * 55)
    print(f"MAE  (Error absoluto medio):              {resultados['mae']:.4f}")
    print(f"RMSE (Raiz del error cuadratico medio):   {resultados['rmse']:.4f}")
    print(f"R²   (Coeficiente de determinacion):      {resultados['r2']:.4f}")
    print("-" * 55)
    print("Coeficientes del Modelo  (Y = β0 + β1·X1 + ... + βn·Xn)")
    print("-" * 55)
    coefs = resultados["coeficientes"]
    print(f"  β0 (intercepto): {coefs['intercepto']:.6f}")
    for variable, valor in coefs["coeficientes"].items():
        print(f"  {variable[:45]:<45}: {valor:.6f}")
    print("-" * 55)


# Selección de modelo
def seleccionar_modelo() -> str:
    print("\n" + "=" * 55)
    print("  Algoritmo Predictivo de Contaminacion del Agua (DQO)")
    print("=" * 55)
    print("  Seleccione el modelo a ejecutar:")
    print("  [1] LSTM (Red Neuronal Recurrente)")
    print("  [2] Regresion Lineal Multivariable")
    print("  [3] Ambos modelos")
    print("=" * 55)

    while True:
        opcion = input("  Opcion: ").strip()
        if opcion in ("1", "2", "3"):
            return opcion
        print("  Opcion invalida. Ingrese 1, 2 o 3.")


def ejecutar_lstm() -> None:
    print("\n[INFO] Iniciando modelo LSTM...")
    lstm = LSTM_Algorithm(CSV, TARGET)
    resultados = lstm.ejecutar()
    mostrar_resultados_lstm(resultados)


def ejecutar_regresion_lineal() -> None:
    print("\n[INFO] Iniciando modelo de Regresion Lineal Multivariable...")
    lr = LinearRegression_Algorithm(CSV, TARGET)
    resultados = lr.ejecutar()
    mostrar_resultados_regresion(resultados)


# Punto de entrada

if __name__ == "__main__":
    opcion = seleccionar_modelo()

    if opcion == "1":
        ejecutar_lstm()
    elif opcion == "2":
        ejecutar_regresion_lineal()