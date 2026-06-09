from metadata import TARGET_COLUMN
from src.source import load_data
from src.transform import Transformer
from src.train import train_model
from src.store import save_model


def main():
    df = load_data("Churn_Modelling_train_test.csv")

    transformer = Transformer()
    df = transformer.transform(df)

    model = train_model(df, TARGET_COLUMN)
    model_path = save_model(model)
    print(f"Model saved in: {model_path}")


if __name__ == "__main__":
    main()
