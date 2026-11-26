import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
MODELS = [
    {"key": "nb_bow", "summary_repeats": 2},
    {"key": "nb_tfidf", "summary_repeats": 2},
    {"key": "lr_bow", "summary_repeats": 1},
    {"key": "lr_tfidf", "summary_repeats": 1},
    {"key": "fnn_tfidf", "summary_repeats": 2},
]


def build_model_input(review_text: str, review_summary: str, summary_repeats: int) -> str:
    summary = (review_summary or "").strip()
    return (f"{summary} " * summary_repeats + review_text.strip()).strip()


def load_expected_metrics() -> dict[str, dict[str, float]]:
    metrics_path = PROJECT_ROOT / "data" / "model_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    if not rows:
        raise ValueError(f"No expected metrics found in {metrics_path}.")

    expected = {}
    for row in rows:
        model_key = row["model_key"]
        if model_key in expected:
            raise ValueError(f"Duplicate metrics row for {model_key}.")
        expected[model_key] = {
            metric: float(row[metric]) for metric in METRIC_NAMES
        }

    configured_keys = {model["key"] for model in MODELS}
    if expected.keys() != configured_keys:
        missing = sorted(configured_keys - expected.keys())
        extra = sorted(expected.keys() - configured_keys)
        raise ValueError(
            f"Metrics CSV model keys do not match configured models "
            f"(missing: {missing}; extra: {extra})."
        )

    return expected


def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    test_path = PROJECT_ROOT / "data" / "UsuableData_20%.xlsx"
    test_data = pd.read_excel(test_path)
    test_data = test_data[["Summary", "Text", "Score"]].copy()
    test_data = test_data.dropna(subset=["Text", "Score"])
    test_data = test_data[test_data["Score"] != 3].copy()
    test_data = test_data.reset_index(drop=True)
    labels = (test_data["Score"] >= 4).astype(int)
    return test_data, labels


def main() -> int:
    expected_metrics = load_expected_metrics()
    test_data, test_labels = load_test_split()
    failures = []

    for config in MODELS:
        model_key = config["key"]
        model_path = PROJECT_ROOT / "models" / f"{model_key}_model.joblib"
        vectorizer_path = (
            PROJECT_ROOT / "models" / f"{model_key}_vectorizer.joblib"
        )
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        model_texts = [
            build_model_input(
                str(text_row.Text),
                "" if pd.isna(text_row.Summary) else str(text_row.Summary),
                config["summary_repeats"],
            )
            for text_row in test_data.itertuples(index=False, name="Review")
        ]
        features = vectorizer.transform(model_texts)
        predictions = model.predict(features)
        actual_metrics = {
            "accuracy": accuracy_score(test_labels, predictions),
            "precision": precision_score(test_labels, predictions, zero_division=0),
            "recall": recall_score(test_labels, predictions, zero_division=0),
            "f1": f1_score(test_labels, predictions, zero_division=0),
        }

        mismatches = []
        for metric, actual in actual_metrics.items():
            expected = expected_metrics[model_key][metric]
            if round(float(actual), 4) != round(expected, 4):
                mismatches.append(
                    f"{metric}: expected {expected:.4f}, got {actual:.4f}"
                )

        if mismatches:
            failures.append(f"{model_key}: " + "; ".join(mismatches))
        else:
            formatted = ", ".join(
                f"{metric}={actual_metrics[metric]:.4f}"
                for metric in METRIC_NAMES
            )
            print(f"PASS {model_key}: {formatted}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print(f"All {len(MODELS)} models match the metrics CSV to 4 decimal places.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
