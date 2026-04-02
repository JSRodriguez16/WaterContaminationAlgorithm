import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data_Manage:

    def __init__(
        self,
        archivo_csv,
        target,
        sequence_length=5,
        coverage_threshold_tabular=0.7,
        coverage_threshold_secuencial=0.5,
        resample_freq_secuencial="QS",
        transformar_target_log=True,
        split_estrategia="temporal",
        random_state=42,
    ):
        self.archivo = archivo_csv
        self.target = target
        self.sequence_length = sequence_length
        self.coverage_threshold_tabular = coverage_threshold_tabular
        self.coverage_threshold_secuencial = coverage_threshold_secuencial
        self.resample_freq_secuencial = resample_freq_secuencial
        self.transformar_target_log = transformar_target_log
        self.split_estrategia = split_estrategia
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        texto_norm = unicodedata.normalize("NFD", str(texto))
        texto_sin_acentos = "".join(
            c for c in texto_norm if unicodedata.category(c) != "Mn"
        )
        return texto_sin_acentos.strip().upper()

    @staticmethod
    def _simplificar_texto(texto: str) -> str:
        texto = Data_Manage._normalizar_texto(texto)
        texto = re.sub(r"\([^\)]*\)", " ", texto)
        texto = re.sub(r"[^A-Z0-9 ]+", " ", texto)
        return re.sub(r"\s+", " ", texto).strip()

    def _resolver_target(self, columnas):
        if self.target in columnas:
            return self.target

        target_normalizado = self._normalizar_texto(self.target)
        target_simple = self._simplificar_texto(self.target)
        columnas_lista = list(columnas)

        for col in columnas_lista:
            if self._normalizar_texto(col) == target_normalizado:
                return col

        for col in columnas_lista:
            if self._simplificar_texto(col) == target_simple:
                return col

        candidatas = [
            col for col in columnas_lista
            if (
                target_simple in self._simplificar_texto(col)
                or self._simplificar_texto(col) in target_simple
            )
        ]

        if len(candidatas) == 1:
            return candidatas[0]
        if len(candidatas) > 1:
            candidatas.sort(key=lambda c: len(self._simplificar_texto(c)))
            return candidatas[0]

        raise KeyError(
            f"No se encontro la variable objetivo '{self.target}' en el dataset."
        )

    @staticmethod
    def _parsear_resultado(serie: pd.Series) -> pd.Series:
        serie = serie.astype(str).str.strip()
        menor_que = serie.str.startswith("<")

        serie = serie.str.replace("<", "", regex=False)
        serie = serie.str.replace(">", "", regex=False)
        serie = serie.str.replace(r"(?<=\d),(?=\d{3}(?:\D|$))", "", regex=True)
        serie = serie.str.replace(",", ".", regex=False)

        valores = pd.to_numeric(serie, errors="coerce")
        valores.loc[menor_que & valores.notna()] *= 0.5
        return valores.where(valores >= 0)

    @staticmethod
    def _extraer_flag_censurado(serie: pd.Series) -> pd.Series:
        return serie.astype(str).str.strip().str.startswith("<").astype(int)

    def _parsear_fechas(self, serie_fechas: pd.Series) -> pd.Series:
        fecha_raw = serie_fechas.astype(str)
        fechas = pd.to_datetime(
            fecha_raw,
            format="%Y %b %d %I:%M:%S %p",
            errors="coerce",
        )

        mask_nat = fechas.isna()
        if mask_nat.any():
            fechas.loc[mask_nat] = pd.to_datetime(fecha_raw[mask_nat], errors="coerce")

        return fechas

    def _filtrar_unidad_dominante(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filtrado = df.copy()
        unidad_col = "UNIDAD DEL RESULTADO"

        if unidad_col not in df_filtrado.columns:
            return df_filtrado

        unidad_limpia = df_filtrado[unidad_col].astype(str).str.strip().str.upper()
        df_filtrado[unidad_col] = unidad_limpia

        dominantes = (
            df_filtrado.groupby("PROPIEDAD OBSERVADA")[unidad_col]
            .agg(lambda s: s.value_counts().index[0] if not s.value_counts().empty else np.nan)
            .to_dict()
        )

        mask_valida = df_filtrado.apply(
            lambda r: (
                pd.isna(dominantes.get(r["PROPIEDAD OBSERVADA"], np.nan))
                or r[unidad_col] == dominantes.get(r["PROPIEDAD OBSERVADA"], np.nan)
            ),
            axis=1,
        )
        return df_filtrado[mask_valida].copy()

    def _cargar_base_long(self) -> pd.DataFrame:
        df = pd.read_csv(self.archivo)
        df.columns = df.columns.str.strip().str.upper()

        df["CENSURADO"] = self._extraer_flag_censurado(df["RESULTADO"])
        df["RESULTADO_NUM"] = self._parsear_resultado(df["RESULTADO"])
        df["FECHA"] = self._parsear_fechas(df["FECHA"])

        df = df.dropna(subset=["FECHA", "RESULTADO_NUM"]).copy()
        df = self._filtrar_unidad_dominante(df)

        return df

    def _long_a_wide(self, df_long: pd.DataFrame) -> pd.DataFrame:
        df_wide = df_long.pivot_table(
            index=["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"],
            columns="PROPIEDAD OBSERVADA",
            values="RESULTADO_NUM",
            aggfunc="median",
        ).reset_index()

        df_cens = df_long.pivot_table(
            index=["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"],
            columns="PROPIEDAD OBSERVADA",
            values="CENSURADO",
            aggfunc="max",
        ).reset_index()

        cols_cens = [
            c
            for c in df_cens.columns
            if c not in ["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"]
        ]
        df_cens = df_cens.rename(columns={c: f"{c}__CENSURADO" for c in cols_cens})

        geo = (
            df_long[
                [
                    "NOMBRE DEL PUNTO DE MONITOREO",
                    "LATITUD",
                    "LONGITUD",
                    "ELEVACIÓN (M.S.N.M.)",
                ]
            ]
            .drop_duplicates()
            .groupby("NOMBRE DEL PUNTO DE MONITOREO", as_index=False)
            .first()
        )

        df_model = df_wide.merge(df_cens, on=["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"], how="left")
        df_model = df_model.merge(geo, on="NOMBRE DEL PUNTO DE MONITOREO", how="left")
        return df_model

    @staticmethod
    def _agregar_features_temporales(df_model: pd.DataFrame) -> pd.DataFrame:
        df_out = df_model.copy()
        df_out["AÑO"] = df_out["FECHA"].dt.year
        df_out["MES"] = df_out["FECHA"].dt.month
        df_out["DIA_DEL_AÑO"] = df_out["FECHA"].dt.dayofyear

        # Variables cíclicas para estacionalidad sin discontinuidad diciembre-enero.
        df_out["MES_SIN"] = np.sin(2 * np.pi * df_out["MES"] / 12.0)
        df_out["MES_COS"] = np.cos(2 * np.pi * df_out["MES"] / 12.0)
        df_out["DIA_SIN"] = np.sin(2 * np.pi * df_out["DIA_DEL_AÑO"] / 365.25)
        df_out["DIA_COS"] = np.cos(2 * np.pi * df_out["DIA_DEL_AÑO"] / 365.25)
        return df_out

    def _regularizar_series_secuenciales(self, df_model: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if not self.resample_freq_secuencial:
            return df_model

        data_cols = [c for c in df_model.columns if c not in ["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"]]
        bloques = []

        for punto, grupo in df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO"):
            g = grupo.sort_values("FECHA").set_index("FECHA")
            if g.index.nunique() < 2:
                g["NOMBRE DEL PUNTO DE MONITOREO"] = punto
                bloques.append(g.reset_index())
                continue

            g_res = g[data_cols].resample(self.resample_freq_secuencial).asfreq()
            g_res["NOMBRE DEL PUNTO DE MONITOREO"] = punto

            if "LATITUD" in g.columns:
                g_res["LATITUD"] = g["LATITUD"].dropna().iloc[0] if g["LATITUD"].notna().any() else np.nan
            if "LONGITUD" in g.columns:
                g_res["LONGITUD"] = g["LONGITUD"].dropna().iloc[0] if g["LONGITUD"].notna().any() else np.nan
            if "ELEVACIÓN (M.S.N.M.)" in g.columns:
                base = "ELEVACIÓN (M.S.N.M.)"
                g_res[base] = g[base].dropna().iloc[0] if g[base].notna().any() else np.nan

            # Distancia temporal al último dato observado para que LSTM/XGBoost capten irregularidad.
            fechas_obs = pd.Series(g.index, index=g.index)
            last_obs = fechas_obs.reindex(g_res.index).ffill()
            g_res["DIAS_DESDE_ULTIMA_MUESTRA"] = (
                (g_res.index.to_series() - last_obs).dt.days
            ).fillna(0)

            bloques.append(g_res.reset_index())

        df_out = pd.concat(bloques, axis=0, ignore_index=True)
        df_out = df_out.sort_values(["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"])
        return df_out.dropna(subset=[target_col]).copy()

    @staticmethod
    def _columnas_modelables(df_model: pd.DataFrame, target_col: str) -> list:
        return [
            c
            for c in df_model.columns
            if c not in ["NOMBRE DEL PUNTO DE MONITOREO", "FECHA", target_col]
        ]

    @staticmethod
    def _crear_indicadores_missing(df_model: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, list]:
        df_out = df_model.copy()
        miss_cols = [f"{col}__MISSING" for col in feature_cols]
        if miss_cols:
            miss_df = pd.DataFrame(
                {f"{col}__MISSING": df_out[col].isna().astype(int) for col in feature_cols},
                index=df_out.index,
            )
            df_out = pd.concat([df_out, miss_df], axis=1)
        return df_out, miss_cols

    @staticmethod
    def _seleccionar_por_cobertura(df_model: pd.DataFrame, feature_cols: list, umbral: float) -> list:
        if not feature_cols:
            return []

        cobertura = (1.0 - df_model[feature_cols].isna().mean()).sort_values(ascending=False)
        seleccion = cobertura[cobertura >= umbral].index.tolist()

        if seleccion:
            return seleccion

        # Fallback para no dejar datasets vacíos con umbrales estrictos.
        return cobertura.head(min(12, len(cobertura))).index.tolist()

    @staticmethod
    def _imputar_sin_fuga(df_model: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df_out = df_model.copy()
        if not feature_cols:
            return df_out

        df_out[feature_cols] = (
            df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO", group_keys=False)[feature_cols]
            .apply(lambda g: g.ffill())
        )

        mediana_punto = df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[feature_cols].transform("median")
        df_out[feature_cols] = df_out[feature_cols].fillna(mediana_punto)

        mediana_global = df_out[feature_cols].median(numeric_only=True)
        df_out[feature_cols] = df_out[feature_cols].fillna(mediana_global)
        df_out[feature_cols] = df_out[feature_cols].fillna(0)
        return df_out

    def _crear_features_tabulares(self, df_model: pd.DataFrame, target_col: str, base_features: list) -> pd.DataFrame:
        df_out = df_model.copy()

        for lag in (1, 2, 3):
            df_out[f"{target_col}__LAG_{lag}"] = df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[target_col].shift(lag)

        top_vars = [c for c in base_features if "__CENSURADO" not in c][:3]
        for col in top_vars:
            df_out[f"{col}__LAG_1"] = df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[col].shift(1)
            df_out[f"{col}__ROLL_MEAN_3"] = (
                df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[col]
                .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            )
            df_out[f"{col}__ROLL_STD_3"] = (
                df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[col]
                .transform(lambda s: s.rolling(window=3, min_periods=2).std())
            )

        return df_out

    def _crear_features_secuenciales(self, df_model: pd.DataFrame, target_col: str, base_features: list) -> pd.DataFrame:
        df_out = df_model.copy()

        # Lags autoregresivos para reforzar señal temporal en LSTM/XGBoost secuencial.
        for lag in (1, 2):
            df_out[f"{target_col}__LAG_{lag}"] = df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[target_col].shift(lag)

        top_vars = [c for c in base_features if "__CENSURADO" not in c][:4]
        for col in top_vars:
            df_out[f"{col}__LAG_1"] = df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[col].shift(1)
            df_out[f"{col}__ROLL_MEAN_3"] = (
                df_out.groupby("NOMBRE DEL PUNTO DE MONITOREO")[col]
                .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            )

        return df_out

    def _cargar_y_limpiar_base(self) -> pd.DataFrame:
        df_long = self._cargar_base_long()
        df_model = self._long_a_wide(df_long)
        df_model = self._agregar_features_temporales(df_model)
        return df_model.sort_values(["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"]).copy()

    def _limpiar_target_dqo(self, df_model: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if "DEMANDA QUIMICA DE OXIGENO" not in self._simplificar_texto(target_col):
            return df_model

        def recortar_iqr(serie: pd.Series, factor: float = 1.5) -> pd.Series:
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr <= 0:
                return serie
            return serie.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)

        df_limpio = df_model.copy()
        df_limpio[target_col] = df_limpio.groupby(
            "NOMBRE DEL PUNTO DE MONITOREO",
            group_keys=False,
        )[target_col].apply(recortar_iqr)

        q_inf = df_limpio[target_col].quantile(0.01)
        q_sup = df_limpio[target_col].quantile(0.99)
        if pd.notna(q_inf) and pd.notna(q_sup) and q_inf < q_sup:
            df_limpio = df_limpio[
                (df_limpio[target_col] >= q_inf) & (df_limpio[target_col] <= q_sup)
            ].copy()

        return df_limpio

    def _preparar_dataset_modelo(self, modo="tabular", regularizar_secuencial=True):
        df_model = self._cargar_y_limpiar_base()
        target_col = self._resolver_target(df_model.columns)

        df_model = df_model.dropna(subset=[target_col]).copy()
        df_model = df_model.sort_values(["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"])

        if modo == "secuencial" and regularizar_secuencial:
            df_model = self._regularizar_series_secuenciales(df_model, target_col)
            df_model = self._agregar_features_temporales(df_model)

        feature_cols = self._columnas_modelables(df_model, target_col)

        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df_model[col]):
                serie = df_model[col].astype(str).str.strip()
                serie = serie.str.replace(r"(?<=\d),(?=\d{3}(?:\D|$))", "", regex=True)
                serie = serie.str.replace(",", ".", regex=False)
                df_model[col] = pd.to_numeric(serie, errors="coerce")

        df_model[feature_cols] = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
        df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")

        df_model = df_model.dropna(subset=[target_col]).copy()
        df_model = self._limpiar_target_dqo(df_model, target_col)

        feature_base = self._columnas_modelables(df_model, target_col)
        umbral = (
            self.coverage_threshold_secuencial
            if modo == "secuencial"
            else self.coverage_threshold_tabular
        )
        feature_seleccionadas = self._seleccionar_por_cobertura(df_model, feature_base, umbral)

        if modo == "tabular":
            df_model = self._crear_features_tabulares(df_model, target_col, feature_seleccionadas)
        if modo == "secuencial":
            df_model = self._crear_features_secuenciales(df_model, target_col, feature_seleccionadas)

        feature_cols = self._columnas_modelables(df_model, target_col)
        df_model, missing_cols = self._crear_indicadores_missing(df_model, feature_cols)
        feature_cols = feature_cols + missing_cols
        df_model = self._imputar_sin_fuga(df_model, feature_cols)

        # En tabular se requiere historial previo para lags válidos.
        if modo == "tabular":
            lags_target = [f"{target_col}__LAG_1", f"{target_col}__LAG_2", f"{target_col}__LAG_3"]
            lags_presentes = [c for c in lags_target if c in df_model.columns]
            if lags_presentes:
                df_model = df_model.dropna(subset=lags_presentes).copy()

        feature_cols = self._columnas_modelables(df_model, target_col)

        return df_model, feature_cols, target_col

    @staticmethod
    def _split_temporal(X, y, train_ratio=0.8):
        split = int(len(X) * train_ratio)
        if split < 1 or split >= len(X):
            raise ValueError("No hay datos suficientes para hacer el split.")
        return X[:split], X[split:], y[:split], y[split:]

    def _split_modelo(self, X, y, train_ratio=0.8):
        if self.split_estrategia == "aleatorio":
            test_size = 1.0 - train_ratio
            if test_size <= 0 or test_size >= 1:
                raise ValueError("train_ratio debe estar entre 0 y 1.")
            return train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=True,
                random_state=self.random_state,
            )
        return self._split_temporal(X, y, train_ratio=train_ratio)

    def _split_temporal_por_punto(self, df_model: pd.DataFrame, feature_cols: list, target_col: str, train_ratio: float):
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        for _, grupo in df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO"):
            g = grupo.sort_values("FECHA")
            if len(g) < 6:
                continue

            X_g = g[feature_cols].to_numpy(dtype=float)
            y_g = g[target_col].to_numpy(dtype=float)

            try:
                X_train_g, X_test_g, y_train_g, y_test_g = self._split_modelo(X_g, y_g, train_ratio)
            except ValueError:
                continue

            X_train_list.append(X_train_g)
            X_test_list.append(X_test_g)
            y_train_list.append(y_train_g)
            y_test_list.append(y_test_g)

        if not X_train_list or not X_test_list:
            raise ValueError("No hay datos suficientes por punto para split temporal supervisado.")

        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        return X_train, X_test, y_train, y_test

    def _transformar_target(self, y: np.ndarray) -> np.ndarray:
        if not self.transformar_target_log:
            return y
        return np.log1p(np.clip(y, a_min=0, a_max=None))

    @staticmethod
    def _crear_secuencias(X, y, sequence_length):
        if len(X) <= sequence_length:
            return np.array([]), np.array([])

        X_seq = [X[i:i + sequence_length] for i in range(len(X) - sequence_length)]
        y_seq = [y[i + sequence_length] for i in range(len(X) - sequence_length)]
        return np.array(X_seq), np.array(y_seq)

    def _armar_dataset_secuencial(self, df_model, feature_cols, target_col, seq_len, train_ratio):
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        for _, grupo in df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO"):
            grupo = grupo.sort_values("FECHA")
            X_g = grupo[feature_cols].to_numpy(dtype=float)
            y_g = grupo[target_col].to_numpy(dtype=float)

            X_seq, y_seq = self._crear_secuencias(X_g, y_g, seq_len)
            if len(X_seq) < 3:
                continue

            X_train_g, X_test_g, y_train_g, y_test_g = self._split_modelo(
                X_seq,
                y_seq,
                train_ratio,
            )

            X_train_list.append(X_train_g)
            X_test_list.append(X_test_g)
            y_train_list.append(y_train_g)
            y_test_list.append(y_test_g)

        if not X_train_list or not X_test_list:
            raise ValueError("No hay secuencias suficientes por punto de monitoreo para entrenar/probar.")

        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        return X_train, X_test, y_train, y_test

    def _escalar_2d(self, X_train, X_test, y_train, y_test):
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _escalar_3d(self, X_train, X_test, y_train, y_test):
        n_steps = X_train.shape[1]
        n_features = X_train.shape[2]

        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)

        self.scaler.fit(X_train_2d)
        X_train_scaled = self.scaler.transform(X_train_2d).reshape(-1, n_steps, n_features)
        X_test_scaled = self.scaler.transform(X_test_2d).reshape(-1, n_steps, n_features)

        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def obtener_dataset_tabular(self):
        return self._preparar_dataset_modelo()

    def preparar_datos_supervisado(self, train_ratio=0.8, escalar=True):
        df_model, feature_cols, target_col = self._preparar_dataset_modelo(modo="tabular")

        X = df_model[feature_cols].to_numpy(dtype=float)
        y = df_model[target_col].to_numpy(dtype=float)
        y = self._transformar_target(y)

        X_train, X_test, y_train, y_test = self._split_temporal_por_punto(
            df_model,
            feature_cols,
            target_col,
            train_ratio,
        )
        y_train = self._transformar_target(y_train)
        y_test = self._transformar_target(y_test)
        if not escalar:
            return X_train, X_test, y_train, y_test
        return self._escalar_2d(X_train, X_test, y_train, y_test)

    def preparar_datos_secuenciales(self, sequence_length=None, train_ratio=0.8, escalar=True):
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        try:
            df_model, feature_cols, target_col = self._preparar_dataset_modelo(
                modo="secuencial",
                regularizar_secuencial=True,
            )
            X_train, X_test, y_train, y_test = self._armar_dataset_secuencial(
                df_model,
                feature_cols,
                target_col,
                seq_len,
                train_ratio,
            )
        except ValueError:
            # Fallback: si el remuestreo reduce demasiado las series, usa la serie original.
            df_model, feature_cols, target_col = self._preparar_dataset_modelo(
                modo="secuencial",
                regularizar_secuencial=False,
            )
            X_train, X_test, y_train, y_test = self._armar_dataset_secuencial(
                df_model,
                feature_cols,
                target_col,
                seq_len,
                train_ratio,
            )

        y_train = self._transformar_target(y_train)
        y_test = self._transformar_target(y_test)

        if not escalar:
            return X_train, X_test, y_train, y_test
        return self._escalar_3d(X_train, X_test, y_train, y_test)

    def desescalar_target(self, y):
        y = np.asarray(y).reshape(-1, 1)
        y_real = self.target_scaler.inverse_transform(y).ravel()
        if self.transformar_target_log:
            y_real = np.expm1(y_real)
        return y_real

    def procesar(self):
        return self.obtener_dataset_tabular()