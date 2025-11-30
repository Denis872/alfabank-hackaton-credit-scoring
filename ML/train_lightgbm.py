import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

RANDOM_STATE = 42

TRAIN_PATH = "/content/drive/My Drive/hackathon_income_train.csv"
TEST_PATH = "/content/drive/My Drive/hackathon_income_test.csv"
FEATURE_DESC_PATH = "features_description.csv"

TARGET_COL = "target"  # имя целевой переменной

DROP_COLS = [
    "id",
    "dt",
    TARGET_COL,
    "w"
]

WEIGHT_COL = "w"  # веса для WMAE

# Явный список категориальных фичей по смыслу
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
    "vert_has_app_ru_raiffeisennews"
]

# =========================
#   ЗАГРУЗКА ДАННЫХ
# =========================

def load_data(path: str) -> pd.DataFrame:
    """
    Унифицированное чтение CSV:
    - разделитель ';'
    - десятичный разделитель ','
    - "nan"/"NaN"/"None"/"" → NaN
    """
    df = pd.read_csv(
        path,
        sep=";",
        decimal=".",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def get_categorical_features(df: pd.DataFrame, feature_desc_path: str | None = None):
    """
    Возвращает список категориальных фич.
    по типу object.
    """
    cat_features = list(df.select_dtypes(include=["object", "category"]).columns)

    cat_features = [c for c in cat_features if c not in DROP_COLS and c != TARGET_COL]
    print(cat_features)
    return cat_features


def prepare_features(
    df: pd.DataFrame,
    cat_features: list[str],
    feature_cols: list[str] | None = None,
):
    """
    Делим на X, y и список категориальных признаков для LGBM.
    """
    assert TARGET_COL in df.columns, f"Не найден target_col={TARGET_COL} в train.csv"

    y = df[TARGET_COL]

    if feature_cols is not None:
        # берём только переданные фичи (после пересечения train/test)
        cols_to_use = [c for c in feature_cols if c in df.columns]
        X = df[cols_to_use].copy()
    else:
        # fallback: старый вариант через DROP_COLS
        X = df.drop(columns=DROP_COLS + [WEIGHT_COL], errors="ignore")

    feature_names = X.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_features if c in feature_names]

    # 👉 выводим первые 5 строк матрицы признаков
    print("\nПервые 5 строк X (матрица признаков):")
    print(X.head())

    return X, y, cat_feature_indices, feature_names


def make_lgb_dataset(X, y, cat_feature_indices, weights=None):
    return lgb.Dataset(
        X,
        label=y,
        weight=weights,
        feature_name=list(X.columns),
        categorical_feature=cat_feature_indices,
        free_raw_data=False,
    )


# =========================
#   КАСТОМНАЯ МЕТРИКА WMAE
# =========================

def lgb_wmae(y_pred: np.ndarray, dataset: lgb.Dataset):
    y_true = dataset.get_label()
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y_true)

    error = np.abs(y_true - y_pred)
    wmae_value = np.sum(w * error) / np.sum(w)
    # False — чем меньше, тем лучше
    return "wmae", wmae_value, False


# =========================
#   ПАРАМЕТРЫ LGBM
# =========================

def get_lgb_params():
    params = {
        # core
        "objective": "regression_l1",  # MAE ближе к WMAE
        "metric": "None",              # метрику считаем кастомной (WMAE)
        "boosting": "gbdt",
        "num_iterations": 10000,       # максимум, будем резать по early stopping
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,

        # регуляризация и контроль переобучения
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 1e-3,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "min_gain_to_split": 0.0,
        "feature_fraction": 0.8,
        "feature_fraction_bynode": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "extra_trees": False,

        # технические
        "force_col_wise": True,   # много фич → col-wise выгоднее
        "num_threads": 0,
        "seed": RANDOM_STATE,
        "data_random_seed": RANDOM_STATE,
        "feature_fraction_seed": RANDOM_STATE,
        "bagging_seed": RANDOM_STATE,

        # уровень логов (аналог verbose в твоём примере)
        "verbose": 1,             # <0: fatal, 0: ошибки, 1: info, >1: debug

        # GPU / CPU
        "device_type": "gpu",     # если нет GPU, поменяй на "cpu"
        "max_bin": 255,
    }
    return params

def get_lgb_param_grid():
    """
    Небольшой осмысленный грид вокруг базовых параметров.
    Можно потом расширить/сузить.
    """
    grid = [
        # базовая конфигурация (что-то близкое к текущей)
        {
            "name": "baseline",
            "learning_rate": 0.03,
            "num_leaves": 64,
            "min_data_in_leaf": 50,
            "lambda_l2": 5.0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "num_iterations": 8000,
        },
        # чуть шире деревья, больше регуляризация, побольше колонок в дереве
        {
            "name": "wide_reg",
            "learning_rate": 0.03,
            "num_leaves": 96,
            "min_data_in_leaf": 40,
            "lambda_l2": 10.0,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "num_iterations": 9000,
        },
        # меньший lr, больше листьев, сильнее L2 — классика
        {
            "name": "lr_002_big_leaves",
            "learning_rate": 0.02,
            "num_leaves": 128,
            "min_data_in_leaf": 40,
            "lambda_l2": 12.0,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "num_iterations": 12000,
        },
        # немного более консервативный вариант
        {
            "name": "lr_002_more_regular",
            "learning_rate": 0.02,
            "num_leaves": 96,
            "min_data_in_leaf": 60,
            "lambda_l2": 15.0,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "num_iterations": 12000,
        },
        # ещё более маленький lr, много деревьев — если данных много, это часто топ
        {
            "name": "lr_0015_deep",
            "learning_rate": 0.015,
            "num_leaves": 160,
            "min_data_in_leaf": 60,
            "lambda_l2": 18.0,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "num_iterations": 15000,
        },
    ]
    return grid


def tune_lgb_params(df: pd.DataFrame, feature_cols: list[str]):
    """
    Перебирает несколько конфигураций LGBM, для каждой считает 5-fold WMAE,
    возвращает лучшие параметры.
    """
    assert TARGET_COL in df.columns, "В train нет target"

    # категориальные фичи
    cat_features = get_categorical_features(df)

    # формируем X, y, веса
    cols_to_use = [c for c in feature_cols if c in df.columns]
    X = df[cols_to_use].copy()
    y = df[TARGET_COL].values
    weights = df[WEIGHT_COL].values

    feature_names = X.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_features if c in feature_names]

    print("\n[ТЮНИНГ] Размер X:", X.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_params = get_lgb_params()
    grid = get_lgb_param_grid()

    best_score = float("inf")
    best_params = None

    for cfg in grid:
        params = base_params.copy()
        params.update(cfg)
        name = params.pop("name", "cfg")

        fold_scores = []

        print(f"\n=== Тестирую конфиг: {name} ===")
        print({k: params[k] for k in ["learning_rate", "num_leaves", "min_data_in_leaf",
                                      "lambda_l2", "feature_fraction", "bagging_fraction",
                                      "num_iterations"]})

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
            w_train = weights[train_idx]
            w_valid = weights[valid_idx]

            dtrain = make_lgb_dataset(X_train, y_train, cat_feature_indices, w_train)
            dvalid = make_lgb_dataset(X_valid, y_valid, cat_feature_indices, w_valid)

            model = lgb.train(
                params,
                train_set=dtrain,
                valid_sets=[dvalid],
                valid_names=["valid"],
                feval=lgb_wmae,
                num_boost_round=params["num_iterations"],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=300,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),  # без спама
                ],
            )

            score = model.best_score["valid"]["wmae"]
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        print(f"[{name}] Mean WMAE: {mean_score:.3f} ± {std_score:.3f}")

        if mean_score < best_score:
            best_score = mean_score
            best_params = params.copy()
            print(f"--> Новый лучший конфиг: {name} (WMAE={best_score:.3f})")

    print("\n=== ЛУЧШИЕ НАЙДЕННЫЕ ПАРАМЕТРЫ ===")
    print(best_params)
    print(f"CV WMAE: {best_score:.3f}")

    return best_params


# =========================
#   ОБУЧЕНИЕ МОДЕЛИ
# =========================

def train_lgb_model(df: pd.DataFrame, feature_cols: list[str] | None = None, params: dict | None = None):
    """
    Обучение финальной модели на лучших параметрах.
    """
    assert TARGET_COL in df.columns, f"Не найден target_col={TARGET_COL} в train.csv"

    if params is None:
        params = get_lgb_params()

    # 1) категориальные фичи
    cat_features = get_categorical_features(df)

    # 2) X, y
    if feature_cols is not None:
        cols_to_use = [c for c in feature_cols if c in df.columns]
        X = df[cols_to_use].copy()
    else:
        X = df.drop(columns=DROP_COLS + [WEIGHT_COL], errors="ignore")

    y = df[TARGET_COL].values
    weights = df[WEIGHT_COL].values

    feature_names = X.columns.tolist()
    cat_feature_indices = [feature_names.index(c) for c in cat_features if c in feature_names]

    print("\n[TRAIN] Размер X:", X.shape)
    print("Кол-во категориальных фичей:", len(cat_feature_indices))

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = []
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold} ===")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        w_train = weights[train_idx]
        w_valid = weights[valid_idx]

        dtrain = make_lgb_dataset(X_train, y_train, cat_feature_indices, w_train)
        dvalid = make_lgb_dataset(X_valid, y_valid, cat_feature_indices, w_valid)

        model = lgb.train(
            params,
            train_set=dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            feval=lgb_wmae,
            num_boost_round=params["num_iterations"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=300,
                    first_metric_only=True,
                    verbose=True,
                ),
                lgb.log_evaluation(period=100),
            ],
        )

        models.append(model)
        best_wmae = model.best_score["valid"]["wmae"]
        fold_scores.append(best_wmae)
        print(f"Fold {fold} best WMAE: {best_wmae:.6f}")

    print(f"\nMean WMAE across folds: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    return models, feature_names, cat_feature_indices

# =========================
#   ПРЕДСКАЗАНИЯ НА TEST
# =========================

def predict_test(models, feature_names, test_path=TEST_PATH):
    test_df = load_data(test_path)
    test_df = clean_object_columns(test_df)

    X_test = test_df[feature_names]

    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict(X_test, num_iteration=model.best_iteration)
    preds /= len(models)

    submission = pd.DataFrame({
        "id": test_df["id"],
        TARGET_COL: preds,
    })

    submission.to_csv("submission.csv", index=False)
    print("submission.csv сохранён")
    return submission


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приведение типов:
    - категории — строго EXPLICIT_CATEGORY_COLS
    - остальные object — пытаемся привести к числам
    """
    df = df.copy()

    for col in df.columns:
        if col in EXPLICIT_CATEGORY_COLS:
            df[col] = df[col].astype("category")
            continue

        if df[col].dtype == "object":
            # пробуем привести колонку к числу
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

            df[col] = s_num  # даже если часть NaN — это ок

    return df


# =========================
#   MAIN
# =========================

def train():
    print("Загружаю hackathon_income_train.csv…")
    train_df = load_data(TRAIN_PATH)

    print("Загружаю hackathon_income_test.csv…")
    test_df = load_data(TEST_PATH)

    train_df = clean_object_columns(train_df)
    test_df = clean_object_columns(test_df)

    # сервисные колонки, которые не должны быть фичами
    service_cols = ["id", "dt", TARGET_COL, WEIGHT_COL]  # у тебя вес называется wmae_weight

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    feature_cols = sorted((train_cols & test_cols) - set(service_cols))

    print("\nКоличество фич:", len(feature_cols))
    print("Первые 20 фич:", feature_cols[:20])

    print("\nТипы первых 20 колонок train:")
    print(train_df.dtypes.head(20))
    print("\nЕсть ли 'target'?", TARGET_COL in train_df.columns)

        # 1) подбираем параметры по CV
    print("\n=== Тюнинг гиперпараметров LightGBM по CV (WMAE) ===")
    best_params = tune_lgb_params(train_df, feature_cols)

    # 2) обучаем финальную модель на лучших параметрах
    print("\n=== Обучаем финальную модель на лучших параметрах ===")
    models, feature_names, cat_feature_indices = train_lgb_model(train_df, feature_cols, params=best_params)


    print("\n=== Обучение завершено ===")
    print("Сохраняю первую модель → model.txt")
    models[0].save_model("model.txt")


if __name__ == "__main__":
    train()
