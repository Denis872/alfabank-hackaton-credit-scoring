# main.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "model.txt"
TEST_PATH = "hackathon_income_test.csv"

EXPLICIT_CATEGORY_COLS = [
    # соц-дем / гео
    "gender",
    "adminarea",
    "city_smart_name",
    "addrref",
    "incomeValueCategory",

    # бинарные флаги
    "blacklist_flag",
    "client_active_flag",
    "nonresident_flag",
    "accountsalary_out_flag",

    # наличие приложений
    "vert_has_app_ru_tinkoff_investing",
    "vert_has_app_ru_vtb_invest",
    "vert_has_app_ru_cian_main",
    "vert_has_app_ru_raiffeisennews",
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        decimal=".",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col in EXPLICIT_CATEGORY_COLS:
            df[col] = df[col].astype("category")
            continue

        if df[col].dtype == "object":
            s = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
            )

            s_num = pd.to_numeric(
                s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )

            df[col] = s_num  

    return df


app = FastAPI(
    title="Hackathon Income Model API",
    description="API для предсказания дохода клиента по id на основе model.txt и test3.csv",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: lgb.Booster | None = None
test_df: pd.DataFrame | None = None
feature_names: list[str] = []


@app.on_event("startup")
def startup_event():
    global model, test_df, feature_names

    print("=== STARTUP ===")
    print(f"Загружаю модель: {MODEL_PATH}")
    model = lgb.Booster(model_file=MODEL_PATH)

    feature_names = model.feature_name()
    print(f"Фич в модели: {len(feature_names)}")

    print(f"Загружаю тестовые данные: {TEST_PATH}")
    df = load_data(TEST_PATH)
    df = clean_object_columns(df)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"В тесте отсутствуют фичи, которые используются в модели: {missing}"
        )

    if "id" not in df.columns:
        raise RuntimeError("В тестовом файле нет колонки 'id'")

    df = df.set_index("id")
    test_df = df

    print("Размер test_df:", test_df.shape)
    print("Первые 5 фич:", feature_names[:5])
    print("=== STARTUP DONE ===")


def predict_for_id(client_id: int) -> float:
    if model is None or test_df is None or not feature_names:
        raise RuntimeError("Модель или данные ещё не инициализированы.")

    if client_id not in test_df.index:
        raise KeyError(f"id={client_id} не найден в test_df")

    row = test_df.loc[client_id].copy()

    for col in feature_names:
        val = row[col]

        # Если строка — заменяем "," на "."
        if isinstance(val, str):
            cleaned = val.replace(",", ".").strip()
            try:
                row[col] = float(cleaned)
            except:
                row[col] = np.nan  

    X = row[feature_names].to_frame().T

    X_np = X.to_numpy()

    preds = model.predict(X_np, num_iteration=model.best_iteration)
    return float(preds[0])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/{client_id}")
def predict(client_id: int):
    try:
        pred = predict_for_id(client_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Клиент с id={client_id} не найден")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")

    return {
        "id": client_id,
        "prediction": pred,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
