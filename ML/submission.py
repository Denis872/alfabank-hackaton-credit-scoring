import numpy as np
import pandas as pd
import lightgbm as lgb

# ==== НАСТРОЙКИ ПУТЕЙ ====
MODEL_PATH = "model.txt"
TEST_PATH = "/content/drive/My Drive/test3.csv"
OUT_PATH = "submission.csv"

# === СПИСОК КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ДОЛЖЕН СОВПАДАТЬ С ОБУЧЕНИЕМ ===
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
    """
    Унифицированное чтение CSV:
    - разделитель ';'
    - десятичный разделитель ','
    """
    df = pd.read_csv(
        path,
        sep=";",
        decimal=".",
        encoding="cp1251",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приведение типов:
    - строго категориальные признаки = EXPLICIT_CATEGORY_COLS
    - остальные object — пытаемся привести к числам
    """
    df = df.copy()

    for col in df.columns:
        # если колонка заранее помечена как категориальная — сразу в category
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

            # пробуем преобразовать в float
            s_num = pd.to_numeric(
                s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )

            df[col] = s_num  # часть значений может стать NaN — это нормально

    return df


def predict(model_path: str, test_path: str, out_path: str):
    print(f"Загружаю модель: {model_path}")
    model = lgb.Booster(model_file=model_path)

    feature_names = model.feature_name()
    print(f"Фич в модели: {len(feature_names)}")

    print(f"Загружаю тестовые данные: {test_path}")
    test_df = load_data(test_path)
    test_df = clean_object_columns(test_df)

    # Проверяем, что все нужные фичи есть в тесте
    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        raise ValueError(
            f"В тесте отсутствуют фичи, которые используются в модели: {missing}"
        )

    # Берём X в том же порядке фич, что и при обучении
    X_test = test_df[feature_names]
    print("Матрица признаков для предсказаний:", X_test.shape)

    print("Считаю предсказания...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    # Если есть колонка id — используем её, иначе просто индекс
    if "id" in test_df.columns:
        ids = test_df["id"]
    else:
        ids = np.arange(len(test_df))

    submission = pd.DataFrame({
        "id": ids,
        "target": preds,
    })

    submission.to_csv(out_path, index=False)
    print(f"Файл с предсказаниями сохранён: {out_path}")
    return submission


# ==== ЗАПУСК В COLAB ====
submission = predict(MODEL_PATH, TEST_PATH, OUT_PATH)
submission.head()
