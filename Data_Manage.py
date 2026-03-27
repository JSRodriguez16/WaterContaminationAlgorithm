import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Data_Manage:

    def __init__(self, archivo_csv, target, sequence_length=5):
        self.archivo = archivo_csv
        self.target = target
        self.sequence_length = sequence_length
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

    def _cargar_y_limpiar_base(self) -> pd.DataFrame:
        df = pd.read_csv(self.archivo)
        df.columns = df.columns.str.strip().str.upper()

        df["RESULTADO"] = self._parsear_resultado(df["RESULTADO"])

        fecha_raw = df["FECHA"].astype(str)
        df["FECHA"] = pd.to_datetime(
            fecha_raw,
            format="%Y %b %d %I:%M:%S %p",
            errors="coerce",
        )

        mask_nat = df["FECHA"].isna()
        if mask_nat.any():
            df.loc[mask_nat, "FECHA"] = pd.to_datetime(fecha_raw[mask_nat], errors="coerce")

        df_wide = df.pivot_table(
            index=["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"],
            columns="PROPIEDAD OBSERVADA",
            values="RESULTADO",
            aggfunc="mean",
        ).reset_index()

        geo = df[
            [
                "NOMBRE DEL PUNTO DE MONITOREO",
                "LATITUD",
                "LONGITUD",
                "ELEVACIÓN (M.S.N.M.)",
            ]
        ].drop_duplicates()

        df_model = df_wide.merge(geo, on="NOMBRE DEL PUNTO DE MONITOREO", how="left")
        df_model["AÑO"] = df_model["FECHA"].dt.year
        df_model["MES"] = df_model["FECHA"].dt.month
        df_model["DIA_DEL_AÑO"] = df_model["FECHA"].dt.dayofyear

        df_model = df_model.sort_values(["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"]).copy()
        cols = [c for c in df_model.columns if c not in ["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"]]
        df_model[cols] = (
            df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO", group_keys=False)[cols]
            .apply(lambda g: g.ffill().bfill())
        )
        return df_model.fillna(0)

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

    def _preparar_dataset_modelo(self):
        df_model = self._cargar_y_limpiar_base()
        target_col = self._resolver_target(df_model.columns)

        df_model = df_model.dropna(subset=[target_col]).copy()
        df_model = df_model.sort_values(["NOMBRE DEL PUNTO DE MONITOREO", "FECHA"])

        feature_cols = [
            c for c in df_model.columns
            if c not in ["NOMBRE DEL PUNTO DE MONITOREO", "FECHA", target_col]
        ]

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
        df_model[feature_cols] = df_model[feature_cols].fillna(0)

        return df_model, feature_cols, target_col

    @staticmethod
    def _split_temporal(X, y, train_ratio=0.8):
        split = int(len(X) * train_ratio)
        if split < 1 or split >= len(X):
            raise ValueError("No hay datos suficientes para hacer el split.")
        return X[:split], X[split:], y[:split], y[split:]

    @staticmethod
    def _crear_secuencias(X, y, sequence_length):
        if len(X) <= sequence_length:
            return np.array([]), np.array([])

        X_seq = [X[i:i + sequence_length] for i in range(len(X) - sequence_length)]
        y_seq = [y[i + sequence_length] for i in range(len(X) - sequence_length)]
        return np.array(X_seq), np.array(y_seq)

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
        df_model, feature_cols, target_col = self._preparar_dataset_modelo()

        X = df_model[feature_cols].to_numpy(dtype=float)
        y = df_model[target_col].to_numpy(dtype=float)

        X_train, X_test, y_train, y_test = self._split_temporal(X, y, train_ratio=train_ratio)
        if not escalar:
            return X_train, X_test, y_train, y_test
        return self._escalar_2d(X_train, X_test, y_train, y_test)

    def preparar_datos_secuenciales(self, sequence_length=None, train_ratio=0.8, escalar=True):
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        df_model, feature_cols, target_col = self._preparar_dataset_modelo()

        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        for _, grupo in df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO"):
            grupo = grupo.sort_values("FECHA")
            X_g = grupo[feature_cols].to_numpy(dtype=float)
            y_g = grupo[target_col].to_numpy(dtype=float)

            X_seq, y_seq = self._crear_secuencias(X_g, y_g, seq_len)
            if len(X_seq) < 3:
                continue

            X_train_g, X_test_g, y_train_g, y_test_g = self._split_temporal(
                X_seq,
                y_seq,
                train_ratio=train_ratio,
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

        if not escalar:
            return X_train, X_test, y_train, y_test
        return self._escalar_3d(X_train, X_test, y_train, y_test)

    def desescalar_target(self, y):
        y = np.asarray(y).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y).ravel()

    def procesar(self):
        return self.obtener_dataset_tabular()
