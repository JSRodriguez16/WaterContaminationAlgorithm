from statistics import linear_regression

from LSTM_Algorithm import LSTM_Algorithm
from LinearRegression_Algorithm import LinearRegression_Algorithm
from XGBoost_Algorithm import XGBoost_Algorithm

# Funciones de visualización de resultados

def mostrar_resultados(nombre_modelo: str, resultados: dict) -> None:
    print(f"\nResultados del Modelo {nombre_modelo}")
    print("-" * 50)
    print(f"  Muestras de entrenamiento: {resultados['n_train']}")
    print(f"  Muestras de prueba: {resultados['n_test']}")
    print("-" * 50)
    print("Precisión del Modelo")
    print("-" * 50)
    print(f"MAE (Error absoluto medio): {resultados['mae']:.4f}")
    print(f"RMSE (Raíz del error cuadrático): {resultados['rmse']:.4f}")
    print(f"R Cuadrado (Coeficiente de determinación): {resultados['r2']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    archivo_datos = "Data_historica_de_calidad_de_agua_20260223.csv"
    variable_objetivo = "DEMANDA QUIMICA DE OXIGENO"

    # --- 1. Ejecución de LSTM ---
    print("\nIniciando entrenamiento de LSTM...")
    lstm = LSTM_Algorithm(archivo_datos, variable_objetivo)
    resultados_lstm = lstm.ejecutar(epochs=30, batch_size=32)
    mostrar_resultados("LSTM", resultados_lstm)

    # --- 2. Ejecución de XGBoost ---
    print("\nIniciando entrenamiento de XGBoost...")
    xgb = XGBoost_Algorithm(archivo_datos, variable_objetivo)
    resultados_xgb = xgb.ejecutar()
    mostrar_resultados("XGBoost", resultados_xgb)
    regresion_lineal = LinearRegression_Algorithm(archivo_datos, variable_objetivo)
    resultados = regresion_lineal.ejecutar()
    mostrar_resultados("Regresion lineal", resultados)