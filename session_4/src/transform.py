import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from metadata import (
    COLUMNS_TO_DROP,
    BINARY_FEATURES,
    ONE_HOT_ENCODE_COLUMNS,
)

class Transformer:
    def __init__(self):
        self.drop_columns = COLUMNS_TO_DROP
        self.binary_variable_columns = BINARY_FEATURES
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df = df.drop(self.drop_columns, axis=1)
        df = self._map_binary_column_to_int(df)
        df = self._one_hot_encoding(df)
        return df

    def _map_binary_column_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0})

        for col in self.binary_variable_columns:
            df[col] = df[col].astype(int)
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
            transform="pandas"
        )
        encoder.fit(df[self.one_hot_encoding_columns])
        encoded_df = encoder.transform(df[self.one_hot_encoding_columns])

        df = df.drop(columns=self.one_hot_encoding_columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df