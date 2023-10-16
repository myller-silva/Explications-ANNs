import pandas as pd


def count_unique_elements(df: pd.DataFrame, column_name: str):
    if column_name in df.columns:
        unique_elements = df[column_name].nunique()
        return unique_elements
    else:
        raise Exception(f"A coluna '{column_name}' não existe no DataFrame.")


