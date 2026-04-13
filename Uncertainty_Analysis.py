import numpy as np


def calcular_metricas_incertidumbre(
    y_train_real: np.ndarray,
    y_pred_train: np.ndarray,
    y_test_real: np.ndarray,
    y_pred_test: np.ndarray,
    z_score: float = 1.96,
) -> dict:
    """Calcula metricas de incertidumbre tipo SUFI-2 para un modelo determinista.

    Aproximacion usada:
    - Se estima la dispersion del error con los residuos de entrenamiento.
    - Se construye un intervalo de prediccion alrededor de y_pred_test.
    - P-factor: cobertura observada dentro del intervalo.
    - R-factor: ancho medio del intervalo normalizado por la desviacion de y observado.
    """
    y_train_real = np.asarray(y_train_real).reshape(-1)
    y_pred_train = np.asarray(y_pred_train).reshape(-1)
    y_test_real = np.asarray(y_test_real).reshape(-1)
    y_pred_test = np.asarray(y_pred_test).reshape(-1)

    residuos_train = y_train_real - y_pred_train
    sigma_residual = float(np.std(residuos_train, ddof=1)) if len(residuos_train) > 1 else 0.0

    half_width = z_score * sigma_residual
    lower = y_pred_test - half_width
    upper = y_pred_test + half_width

    en_intervalo = (y_test_real >= lower) & (y_test_real <= upper)
    p_factor = float(np.mean(en_intervalo)) if len(y_test_real) else 0.0

    std_obs = float(np.std(y_test_real, ddof=1)) if len(y_test_real) > 1 else 0.0
    mean_width = float(np.mean(upper - lower)) if len(y_test_real) else 0.0
    r_factor = mean_width / std_obs if std_obs > 1e-12 else float("nan")

    return {
        "sigma_residual_train": sigma_residual,
        "uncertainty_zscore": z_score,
        "interval_lower": lower,
        "interval_upper": upper,
        "p_factor": p_factor,
        "r_factor": r_factor,
        "mean_prediction_interval_width": mean_width,
    }
