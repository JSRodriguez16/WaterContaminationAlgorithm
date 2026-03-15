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

        df["RESULTADO"] = df["RESULTADO"].astype(str)

        df["RESULTADO"] = df["RESULTADO"].str.replace("<", "", regex=False)
        df["RESULTADO"] = df["RESULTADO"].str.replace(">", "", regex=False)

        df["RESULTADO"] = pd.to_numeric(df["RESULTADO"], errors="coerce")

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
        df_model[feature_cols] = df_model[feature_cols].fillna(0)

        return df_model, feature_cols, target_col

    def _crear_secuencias_por_punto(self, df_model, feature_cols, target_col):

        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        for _, grupo in df_model.groupby("NOMBRE DEL PUNTO DE MONITOREO"):
            grupo = grupo.sort_values("FECHA")
            X_g = grupo[feature_cols].to_numpy(dtype=float)
            y_g = grupo[target_col].to_numpy(dtype=float)

            if len(X_g) <= self.sequence_length:
                continue

            X_seq, y_seq = [], []
            for i in range(len(X_g) - self.sequence_length):
                X_seq.append(X_g[i:i + self.sequence_length])
                y_seq.append(y_g[i + self.sequence_length])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            if len(X_seq) < 3:
                continue

            split = int(len(X_seq) * 0.8)
            if split < 1 or split >= len(X_seq):
                continue

            X_train_list.append(X_seq[:split])
            y_train_list.append(y_seq[:split])
            X_test_list.append(X_seq[split:])
            y_test_list.append(y_seq[split:])

        if not X_train_list or not X_test_list:
            raise ValueError(
                "No hay secuencias suficientes por punto de monitoreo para entrenar/probar."
            )

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        return X_train, X_test, y_train, y_test

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

    def desescalar_target(self, y):
        y = np.asarray(y).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y).ravel()

    def procesar(self):

        df = self._cargar_datos()

        df = self._limpiar_resultados(df)

        df_wide = self._pivot_vertical(df)

        df_wide = self._crear_variables_contextuales(df_wide, df)

        df_wide = self._manejar_nan(df_wide)

        df_model, feature_cols, target_col = self._preparar_features(df_wide)

        X_train, X_test, y_train, y_test = self._crear_secuencias_por_punto(
            df_model,
            feature_cols,
            target_col,
        )

        return self._escalar_train_test(X_train, X_test, y_train, y_test)