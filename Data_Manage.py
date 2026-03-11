import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class PreprocesamientoAgua:

    def __init__(self, archivo_csv, target, sequence_length=5):
        self.archivo = archivo_csv
        self.target = target
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def _cargar_datos(self):

        df = pd.read_csv(self.archivo)
        df.columns = df.columns.str.strip().str.upper()

        return df

    def _limpiar_resultados(self, df):

        df["RESULTADO"] = df["RESULTADO"].astype(str)

        df["RESULTADO"] = df["RESULTADO"].str.replace("<", "", regex=False)
        df["RESULTADO"] = df["RESULTADO"].str.replace(">", "", regex=False)

        df["RESULTADO"] = pd.to_numeric(df["RESULTADO"], errors="coerce")

        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

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

        df = df.ffill()
        df = df.bfill()
        df = df.fillna(0)

        return df

    def _preparar_features(self, df):

        df = df.dropna(subset=[self.target])

        X = df.drop(columns=[
            "NOMBRE DEL PUNTO DE MONITOREO",
            "FECHA",
            self.target
        ])

        y = df[self.target]

        X = self.scaler.fit_transform(X)

        return X, y.values

    def _crear_secuencias(self, X, y):

        X_seq = []
        y_seq = []

        for i in range(len(X) - self.sequence_length):

            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def procesar(self):

        df = self._cargar_datos()

        df = self._limpiar_resultados(df)

        df_wide = self._pivot_vertical(df)

        df_wide = self._crear_variables_contextuales(df_wide, df)

        df_wide = self._manejar_nan(df_wide)

        X, y = self._preparar_features(df_wide)

        X_seq, y_seq = self._crear_secuencias(X, y)

        if len(X_seq) == 0:
            raise ValueError(
                "No se pudieron crear secuencias. "
                "Reduce sequence_length o verifica el volumen de datos."
            )

        split = int(len(X_seq) * 0.8)

        X_train = X_seq[:split]
        X_test = X_seq[split:]

        y_train = y_seq[:split]
        y_test = y_seq[split:]

        return X_train, X_test, y_train, y_test