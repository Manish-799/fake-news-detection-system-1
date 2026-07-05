import json
import pickle
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from backend.text_utils import clean_text


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

TRUE_DATASET_PATH = DATASET_DIR / "True.csv"
FAKE_DATASET_PATH = DATASET_DIR / "Fake.csv"

MODEL_PATH = ARTIFACTS_DIR / "model_final.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer_final.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# ---------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not TRUE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"True news dataset not found at: "
            f"{TRUE_DATASET_PATH}"
        )

    if not FAKE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Fake news dataset not found at: "
            f"{FAKE_DATASET_PATH}"
        )

    print("Loading datasets...")

    true_news = pd.read_csv(TRUE_DATASET_PATH)
    fake_news = pd.read_csv(FAKE_DATASET_PATH)

    required_columns = {"title", "text"}

    for dataset_name, dataset in [
        ("True.csv", true_news),
        ("Fake.csv", fake_news),
    ]:
        missing_columns = required_columns - set(dataset.columns)

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

    true_news["label"] = 1
    fake_news["label"] = 0

    # Combine title and article body
    true_news["content"] = (
        true_news["title"].fillna("").astype(str)
        + " "
        + true_news["text"].fillna("").astype(str)
    )

    fake_news["content"] = (
        fake_news["title"].fillna("").astype(str)
        + " "
        + fake_news["text"].fillna("").astype(str)
    )

    data = pd.concat(
        [
            true_news[["content", "label"]],
            fake_news[["content", "label"]],
        ],
        ignore_index=True,
    )

    # Used for blank detection and duplicate removal
    data["normalized_content"] = (
        data["content"].apply(clean_text)
    )

    data = data[
        data["normalized_content"].str.len() > 0
    ]

    rows_before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=["normalized_content"]
    )

    duplicate_count = rows_before_duplicates - len(data)

    data = data.drop(
        columns=["normalized_content"]
    )

    data = data.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    print(f"Total usable articles: {len(data)}")
    print(f"Duplicate articles removed: {duplicate_count}")

    print("\nDataset balance:")
    print(data["label"].value_counts())

    return data


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train_model(data: pd.DataFrame):
    X = data["content"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        max_df=0.8,
        min_df=5,
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    print("\nFitting TF-IDF vectorizer...")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
    )

    print("\nTraining Logistic Regression model...")

    model.fit(X_train_vec, y_train)

    return (
        model,
        vectorizer,
        X_test_vec,
        y_test,
        len(X_train),
        len(X_test),
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    model,
    vectorizer,
    X_test_vec,
    y_test,
    train_rows: int,
    test_rows: int,
):
    predictions = model.predict(X_test_vec)

    probabilities = model.predict_proba(X_test_vec)

    class_to_index = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    real_probabilities = probabilities[
        :,
        class_to_index[1],
    ]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        real_probabilities,
    )

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print("Rows = Actual, Columns = Predicted")
    print("[Fake, Real]")
    print(confusion)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "vocabulary_size": len(
            vectorizer.vocabulary_
        ),
    }

    return metrics


# ---------------------------------------------------------
# FEATURE ANALYSIS
# ---------------------------------------------------------

def display_feature_analysis(
    model,
    vectorizer,
    top_n: int = 15,
):
    feature_names = (
        vectorizer.get_feature_names_out()
    )

    coefficients = model.coef_[0]

    real_indices = coefficients.argsort()[
        -top_n:
    ][::-1]

    fake_indices = coefficients.argsort()[
        :top_n
    ]

    print("\n" + "=" * 60)
    print("GLOBAL FEATURE ANALYSIS")
    print("=" * 60)

    print("\nTop REAL news indicators:")

    for index in real_indices:
        print(
            f"{feature_names[index]:30} "
            f"{coefficients[index]:.4f}"
        )

    print("\nTop FAKE news indicators:")

    for index in fake_indices:
        print(
            f"{feature_names[index]:30} "
            f"{coefficients[index]:.4f}"
        )


# ---------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------

def save_artifacts(
    model,
    vectorizer,
    metrics: dict,
):
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, "wb") as file:
        pickle.dump(vectorizer, file)

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("ARTIFACTS SAVED")
    print("=" * 60)

    print(f"Model:      {MODEL_PATH}")
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Metrics:    {METRICS_PATH}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    data = load_dataset()

    (
        model,
        vectorizer,
        X_test_vec,
        y_test,
        train_rows,
        test_rows,
    ) = train_model(data)

    metrics = evaluate_model(
        model=model,
        vectorizer=vectorizer,
        X_test_vec=X_test_vec,
        y_test=y_test,
        train_rows=train_rows,
        test_rows=test_rows,
    )

    display_feature_analysis(
        model=model,
        vectorizer=vectorizer,
    )

    save_artifacts(
        model=model,
        vectorizer=vectorizer,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()