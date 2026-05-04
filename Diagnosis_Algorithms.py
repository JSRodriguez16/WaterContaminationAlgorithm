from LSTM_Algorithm import LSTM_Algorithm
from LinearRegression_Algorithm import LinearRegression_Algorithm
from XGBoost_Algorithm import XGBoost_Algorithm
from SVM_Algorithm import SVM_Algorithm
import time

def mostrar_resultados(nombre_modelo: str, resultados: dict) -> None:
    print(f"\nResultados del Modelo {nombre_modelo}")
    print("-" * 50)
    print(f"  Muestras de entrenamiento: {resultados['n_train']}")
    print(f"  Muestras de prueba: {resultados['n_test']}")
    print("-" * 50)
    print("Precision del Modelo")
    print("-" * 50)
    if "mae_train" in resultados and "rmse_train" in resultados:
        print(f"MAE entrenamiento: {resultados['mae_train']:.4f}")
        print(f"RMSE entrenamiento: {resultados['rmse_train']:.4f}")
    print(f"MAE prueba: {resultados['mae']:.4f}")
    print(f"RMSE prueba: {resultados['rmse']:.4f}")
    print(f"R Cuadrado (Coeficiente de determinacion): {resultados['r2']:.4f}")
    if resultados.get("shap_plot_mode") == "interactivo":
        print("Grafico SHAP mostrado en ventana interactiva de matplotlib.")
    if resultados.get("shap_plot_path"):
        print(f"Grafico SHAP guardado en: {resultados['shap_plot_path']}")
    if resultados.get("shap_feature_importance"):
        print("Importancia global SHAP (top variables):")
        for item in resultados["shap_feature_importance"][:10]:
            print(f"  - {item['feature']}: {item['importance']:.6f}")
    if resultados.get("svm_mejores_hiperparametros"):
        print("-" * 50)
        print("Diagnostico SVM")
        print("-" * 50)
        print(f"  Kernel: {resultados.get('svm_kernel', 'rbf')}")
        print(f"  Mejores hiperparametros: {resultados['svm_mejores_hiperparametros']}")
        if resultados.get("svm_mejor_rmse_cv") is not None:
            print(f"  Mejor RMSE CV (busqueda): {resultados['svm_mejor_rmse_cv']:.4f}")
        if resultados.get("svm_n_support_vectors") is not None:
            print(f"  Vectores de soporte: {resultados['svm_n_support_vectors']}")
    if resultados.get("svm_importancia_variables"):
        print("  Importancia por permutacion (top variables):")
        for item in resultados["svm_importancia_variables"][:10]:
            print(f"    - {item['feature']}: {item['importance']:.6f}")
    if "p_factor" in resultados and "r_factor" in resultados:
        print("-" * 50)
        print("Analisis de Incertidumbre")
        print("-" * 50)
        print(f"P-factor (cobertura): {resultados['p_factor']:.4f}")
        print(f"R-factor (ancho relativo): {resultados['r_factor']:.4f}")
        print(
            "Ancho medio del intervalo 95%: "
            f"{resultados['mean_prediction_interval_width']:.4f}"
        )
        print(
            "Sigma residual (train): "
            f"{resultados['sigma_residual_train']:.4f}"
        )
    print("-" * 50)

if __name__ == "__main__":
    archivo_datos = "Data_historica_de_calidad_de_agua_20260223.csv"
    variable_objetivo = "DEMANDA QUIMICA DE OXIGENO"

    print("\n" + "="*60)
    print("Entrenamiento y Evaluacion de Modelos")
    print("="*60)

    print("\nIniciando entrenamiento de LSTM")
    inicio_lstm = time.time()
    lstm = LSTM_Algorithm(archivo_datos, variable_objetivo)
    resultados_lstm = lstm.ejecutar(epochs=80, batch_size=32)
    tiempo_lstm = time.time() - inicio_lstm
    mostrar_resultados("LSTM", resultados_lstm)
    print(f"Tiempo LSTM: {tiempo_lstm:.2f}s")

    print("\nIniciando entrenamiento de XGBoost")
    inicio_xgb = time.time()
    xgb = XGBoost_Algorithm(archivo_datos, variable_objetivo)
    resultados_xgb = xgb.ejecutar()
    tiempo_xgb = time.time() - inicio_xgb
    mostrar_resultados("XGBoost", resultados_xgb)
    print(f"Tiempo XGBoost: {tiempo_xgb:.2f}s")

    print("\nIniciando entrenamiento de SVM")
    inicio_svm = time.time()
    svm = SVM_Algorithm(archivo_datos, variable_objetivo)
    resultados_svm = svm.ejecutar()
    tiempo_svm = time.time() - inicio_svm
    mostrar_resultados("SVM", resultados_svm)
    print(f"Tiempo SVM: {tiempo_svm:.2f}s")

    print("\nIniciando entrenamiento de Regresion Lineal")
    inicio_reg = time.time()
    regresion_lineal = LinearRegression_Algorithm(archivo_datos, variable_objetivo)
    resultados = regresion_lineal.ejecutar()
    tiempo_reg = time.time() - inicio_reg
    mostrar_resultados("Regresion lineal", resultados)
    print(f"Tiempo Regresion Lineal: {tiempo_reg:.2f}s")