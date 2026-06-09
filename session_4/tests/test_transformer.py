import pandas as pd

from src.transform import Transformer


def test_transformer_drops_columns():
    df = pd.DataFrame(
        {
            "RowNumber": [1],
            "CustomerId": [123],
            "Surname": ["Smith"],
            "CreditScore": [650],
            "Geography": ["Spain"],
            "Gender": ["Female"],
            "Age": [35],
            "Tenure": [3],
            "Balance": [1000.0],
            "NumOfProducts": [2],
            "HasCrCard": [1],
            "IsActiveMember": [1],
            "EstimatedSalary": [50000.0],
            "Exited": [0],
        }
    )

    transformer = Transformer()
    transformed_df = transformer.transform(df)

    assert "RowNumber" not in transformed_df.columns
    assert "CustomerId" not in transformed_df.columns
    assert "Surname" not in transformed_df.columns


def test_transformer_converts_gender_to_int():
    df = pd.DataFrame(
        {
            "RowNumber": [1],
            "CustomerId": [123],
            "Surname": ["Smith"],
            "CreditScore": [650],
            "Geography": ["Spain"],
            "Gender": ["Female"],
            "Age": [35],
            "Tenure": [3],
            "Balance": [1000.0],
            "NumOfProducts": [2],
            "HasCrCard": [1],
            "IsActiveMember": [1],
            "EstimatedSalary": [50000.0],
            "Exited": [0],
        }
    )

    transformer = Transformer()
    transformed_df = transformer.transform(df)

    assert transformed_df["Gender"].iloc[0] == 1


def test_transformer_encodes_geography():
    df = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [123, 456],
            "Surname": ["Smith", "Jones"],
            "CreditScore": [650, 700],
            "Geography": ["Spain", "Germany"],
            "Gender": ["Female", "Male"],
            "Age": [35, 40],
            "Tenure": [3, 5],
            "Balance": [1000.0, 2000.0],
            "NumOfProducts": [2, 1],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 0],
            "EstimatedSalary": [50000.0, 60000.0],
            "Exited": [0, 1],
        }
    )

    transformer = Transformer()
    transformed_df = transformer.transform(df)

    assert "Geography" not in transformed_df.columns
    assert "Geography_Spain" in transformed_df.columns
