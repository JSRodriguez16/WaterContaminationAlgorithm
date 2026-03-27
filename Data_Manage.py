import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.preprocessing import StandardScaler


class PreprocesamientoAgua:

    def __init__(self, archivo_csv, target, sequence_length=5):
        self.archivo = archivo_csv
        self.target = target
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def _cargar_datos(self):

        df = pd.read_csv(self.archivo)
        df.columns = df.columns.str.strip().str.upper()

        return df

    def _limpiar_resultados(self, df):
        serie = df["RESULTADO"].astype(str).str.strip()

        # Para datos censurados tipo <x, se usa x/2 como aproximacion comun en analisis ambiental.
        es_menor_que = serie.str.startswith("<")

        serie = serie.str.replace("<", "", regex=False)
        serie = serie.str.replace(">", "", regex=False)
        serie = serie.str.replace(r"(?<=\d),(?=\d{3}(?:\D|$))", "", regex=True)
        serie = serie.str.replace(",", ".", regex=False)

        valores = pd.to_numeric(serie, errors="coerce")
        valores.loc[es_menor_que & valores.notna()] = (
            valores.loc[es_menor_que & valores.notna()] * 0.5
        )
        valores = valores.where(valores >= 0)
        df["RESULTADO"] = valores

        fecha_raw = df["FECHA"].astype(str)

        df["FECHA"] = pd.to_datetime(
            fecha_raw,
            format="%Y %b %d %I:%M:%S %p",
            errors="coerce"
        )

        mask_nat = df["FECHA"].isna()
        if mask_nat.any():
            df.loc[mask_nat, "FECHA"] = pd.to_datetime(
                fecha_raw[mask_nat],
                errors="coerce"
            )

        return df

    def _es_target_dqo(self, target_col: str) -> bool:
        target_simple = self._simplificar_texto(target_col)
        return "DEMANDA QUIMICA DE OXIGENO" in target_simple or target_simple == "DQO"

    @staticmethod
    def _recortar_outliers_iqr(serie: pd.Series, factor: float = 1.5) -> pd.Series:
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            return serie
        limite_inf = q1 - factor * iqr
        limite_sup = q3 + factor * iqr
        return serie.clip(lower=limite_inf, upper=limite_sup)

    def _limpiar_target_dqo(self, df_model: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if not self._es_target_dqo(target_col):
            return df_model

        df_limpio = df_model.copy()

        df_limpio[target_col] = df_limpio.groupby(
            "NOMBRE DEL PUNTO DE MONITOREO",
            group_keys=False,
        )[target_col].apply(lambda s: self._recortar_outliers_iqr(s, factor=1.5))

        q_inf = df_limpio[target_col].quantile(0.01)
        q_sup = df_limpio[target_col].quantile(0.99)
        if pd.notna(q_inf) and pd.notna(q_sup) and q_inf < q_sup:
            df_limpio = df_limpio[
                (df_limpio[target_col] >= q_inf) & (df_limpio[target_col] <= q_sup)
            ].copy()

        return df_limpio

    def _pivot_vertical(self, df):

        df_wide = df.pivot_table(
            index=["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"],
            columns="PROPIEDAD OBSERVADA",
            values="RESULTADO",
            aggfunc="mean"
        ).reset_index()

        return df_wide

    def _crear_variables_contextuales(self, df_wide, df_original):

        geo = df_original[
            [
                "NOMBRE DEL PUNTO DE MONITOREO",
                "LATITUD",
                "LONGITUD",
                "ELEVACIÓN (M.S.N.M.)"
            ]
        ].drop_duplicates()

        df_contexto = df_wide.merge(
            geo,
            on="NOMBRE DEL PUNTO DE MONITOREO",
            how="left"
        )

        df_contexto["AÑO"] = df_contexto["FECHA"].dt.year
        df_contexto["MES"] = df_contexto["FECHA"].dt.month
        df_contexto["DIA_DEL_AÑO"] = df_contexto["FECHA"].dt.dayofyear

        return df_contexto

    def _manejar_nan(self, df):
        df = df.sort_values([
            "NOMBRE DEL PUNTO DE MONITOREO",
            "FECHA"
        ]).copy()

        columnas_grupo = [c for c in df.columns if c not in ["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"]]
        df[columnas_grupo] = (
            df.groupby("NOMBRE DEL PUNTO DE MONITOREO", group_keys=False)[columnas_grupo]
            .apply(lambda g: g.ffill().bfill())
        )

        df = df.fillna(0)
        return df

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        texto_norm = unicodedata.normalize("NFD", str(texto))
        texto_sin_acentos = "".join(
            c for c in texto_norm if unicodedata.category(c) != "Mn"
        )
        return texto_sin_acentos.strip().upper()

    @staticmethod
    def _simplificar_texto(texto: str) -> str:
        texto = PreprocesamientoAgua._normalizar_texto(texto)
        texto = re.sub(r"\([^\)]*\)", " ", texto)
        texto = re.sub(r"[^A-Z0-9 ]+", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

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

        candidatas = []
        for col in columnas_lista:
            col_simple = self._simplificar_texto(col)
            if target_simple in col_simple or col_simple in target_simple:
                candidatas.append(col)

        if len(candidatas) == 1:
            return candidatas[0]
        if len(candidatas) > 1:
            candidatas.sort(key=lambda c: len(self._simplificar_texto(c)))
            return candidatas[0]

        raise KeyError(
            f"No se encontró la variable objetivo '{self.target}' en el dataset."
        )

    def _preparar_features(self, df):
        target_col = self._resolver_target(df.columns)

        df_model = df.dropna(subset=[target_col]).copy()
        df_model = df_model.sort_values([
            "NOMBRE DEL PUNTO DE MONITOREO",
            "FECHA",
        ])

        feature_cols = [c for c in df_model.columns if c not in [
            "NOMBRE DEL PUNTO DE MONITOREO",
            "FECHA",
            target_col,
        ]]

        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df_model[col]):
                serie = df_model[col].astype(str).str.strip()
                serie = serie.str.replace(r"(?<=\d),(?=\d{3}(?:\D|$))", "", regex=True)
                serie = serie.str.replace(",", ".", regex=False)
                df_model[col] = pd.to_numeric(serie, errors="coerce")

        for col in feature_cols:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

        df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")

        df_model = df_model.dropna(subset=[target_col]).copy()
        df_model = self._limpiar_target_dqo(df_model, target_col)
        df_model[feature_cols] = df_model[feature_cols].fillna(0)

        return df_model, feature_cols, target_col

    def _escalar_train_test(self, X_train, X_test, y_train, y_test):

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

    def _crear_secuencias(self, X, y, sequence_length):
        if len(X) <= sequence_length:
            return np.array([]), np.array([])

        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def _dividir_train_test(self, X, y, train_ratio=0.8):
        split = int(len(X) * train_ratio)
        if split < 1 or split >= len(X):
            raise ValueError("No hay datos suficientes para hacer el split.")
        return X[:split], X[split:], y[:split], y[split:]

    def obtener_dataset_tabular(self):
        df = self._cargar_datos()
        df = self._limpiar_resultados(df)
        df_wide = self._pivot_vertical(df)
        df_wide = self._crear_variables_contextuales(df_wide, df)
        df_wide = self._manejar_nan(df_wide)
        df_model, feature_cols, target_col = self._preparar_features(df_wide)
        return df_model, feature_cols, target_col

    def preparar_datos_supervisado(self, train_ratio=0.8, escalar=True):
        df_model, feature_cols, target_col = self.obtener_dataset_tabular()

        X = df_model[feature_cols].to_numpy(dtype=float)
        y = df_model[target_col].to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = self._dividir_train_test(
            X,
            y,
            train_ratio=train_ratio,
        )

        if not escalar:
            return X_train, X_test, y_train, y_test

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def preparar_datos_secuenciales(self, sequence_length=None, train_ratio=0.8, escalar=True):
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        df_model, feature_cols, target_col = self.obtener_dataset_tabular()

        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        for _, grupo in df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO"):
            grupo = grupo.sort_values("FECHA")
            X_g = grupo[feature_cols].to_numpy(dtype=float)
            y_g = grupo[target_col].to_numpy(dtype=float)

            X_seq, y_seq = self._crear_secuencias(X_g, y_g, seq_len)
            if len(X_seq) < 3:
                continue

            X_train_g, X_test_g, y_train_g, y_test_g = self._dividir_train_test(
                X_seq,
                y_seq,
                train_ratio=train_ratio,
            )

            X_train_list.append(X_train_g)
            X_test_list.append(X_test_g)
            y_train_list.append(y_train_g)
            y_test_list.append(y_test_g)

        if not X_train_list or not X_test_list:
            raise ValueError(
                "No hay secuencias suficientes por punto de monitoreo."
            )

        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        if not escalar:
            return X_train, X_test, y_train, y_test

        return self._escalar_train_test(X_train, X_test, y_train, y_test)

    def desescalar_target(self, y):
        y = np.asarray(y).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y).ravel()

    def procesar(self):
        return self.obtener_dataset_tabular()