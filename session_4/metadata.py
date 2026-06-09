MODELS_FOLDER = "session_4/models"
DATASETS_FOLDER = "session_4/datasets"
MODEL_NAME = "class_model-Marah"

TARGET_COLUMN = "Exited"

COLUMNS_TO_DROP = [
    "RowNumber",
    "CustomerId",
    "Surname",
]

BINARY_FEATURES = [
    "Gender",
    "HasCrCard",
    "IsActiveMember",
]

ONE_HOT_ENCODE_COLUMNS = [
    "Geography",
]

MODEL_PARAMS = {"max_depth": 5, "random_state": 42}
