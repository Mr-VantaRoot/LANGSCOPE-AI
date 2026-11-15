"""Train the nine-language Tatoeba demo using train/validation only.

The existing Tatoeba test set was already evaluated by an earlier ten-label
experiment. This script deliberately does not score it. A fresh test set is
required before publishing final evaluation metrics for this nine-label model.
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from phase_tatoeba import LABELS, build_splits


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "tatoeba_nine"
SEED = 42
REGULARIZATION_VALUES = [0.1, 1.0, 10.0]
REMOVED_LABEL = "Afaan Oromo (West Central Oromo)"


def main():
    train_rows, validation_rows, _previous_test_rows, split_counts = build_splits()
    labels = [label for label in LABELS if label != REMOVED_LABEL]
    train_rows = [row for row in train_rows if row["label"] in labels]
    validation_rows = [row for row in validation_rows if row["label"] in labels]

    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]
    validation_labels = [row["label"] for row in validation_rows]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    train_features = vectorizer.fit_transform(train_texts)
    validation_features = vectorizer.transform(validation_texts)

    best_key = None
    best_classifier = None
    validation_grid = []
    for c_value in REGULARIZATION_VALUES:
        classifier = LogisticRegression(
            C=c_value,
            max_iter=2000,
            random_state=SEED,
        )
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(validation_features).tolist()
        accuracy = float(accuracy_score(validation_labels, predictions))
        macro_f1 = float(
            f1_score(
                validation_labels,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        validation_grid.append(
            {"C": c_value, "accuracy": accuracy, "macro_f1": macro_f1}
        )
        print(
            f"C={c_value}: validation accuracy={accuracy:.6f}; "
            f"macro-F1={macro_f1:.6f}"
        )
        key = (macro_f1, accuracy)
        if best_key is None or key > best_key:
            best_key = key
            best_classifier = classifier

    # Calibrate the app's uncertainty threshold on validation data only.
    validation_probabilities = best_classifier.predict_proba(validation_features)
    confidences = validation_probabilities.max(axis=1)
    predictions = best_classifier.classes_[validation_probabilities.argmax(axis=1)]
    best_threshold = None
    best_threshold_key = None
    threshold_result = None
    for threshold in sorted(set(float(value) for value in confidences)):
        accepted = confidences >= threshold
        coverage = float(accepted.mean())
        if coverage < 0.95:
            continue
        correct = sum(
            prediction == truth
            for prediction, truth, keep in zip(
                predictions, validation_labels, accepted
            )
            if keep
        )
        selective_accuracy = correct / int(accepted.sum())
        key = (selective_accuracy, threshold)
        if best_threshold_key is None or key > best_threshold_key:
            best_threshold_key = key
            best_threshold = threshold
            threshold_result = {
                "threshold": threshold,
                "validation_coverage": coverage,
                "validation_selective_accuracy": selective_accuracy,
                "validation_examples": len(validation_labels),
                "minimum_coverage": 0.95,
                "test_split_used": False,
            }

    if best_threshold is None:
        raise RuntimeError("No validation threshold met 95% coverage")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, OUTPUT_DIR / "vectorizer.joblib")
    joblib.dump(best_classifier, OUTPUT_DIR / "classifier.joblib")
    (OUTPUT_DIR / "uncertainty_threshold.json").write_text(
        json.dumps(threshold_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset": "Tatoeba per-language sentence exports",
        "languages": labels,
        "removed_from_scope": REMOVED_LABEL,
        "seed": SEED,
        "training_source": "Tatoeba only",
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_per_language": {
            label: sum(row["label"] == label for row in train_rows)
            for label in labels
        },
        "validation_per_language": {
            label: sum(row["label"] == label for row in validation_rows)
            for label in labels
        },
        "previous_test_set_scored": False,
        "fresh_test_set_required": True,
        "model": {
            "vectorizer": "TfidfVectorizer(analyzer='char', ngram_range=(1, 3))",
            "classifier": "LogisticRegression",
            "selected_C": best_classifier.C,
            "validation_grid": validation_grid,
            "selection_metric": "validation macro-F1, then validation accuracy",
        },
        "threshold": threshold_result,
        "archived_split_counts_reference": split_counts["after_balancing"],
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_rows = [
        next(row for row in train_rows if row["label"] == label)
        for label in labels
    ]
    (OUTPUT_DIR / "training_examples.json").write_text(
        json.dumps(sample_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected C={best_classifier.C}; validation threshold={best_threshold:.6f}")
    print(f"Saved nine-language model to {OUTPUT_DIR}")
    print("No held-out test predictions or evaluation metrics were produced.")


if __name__ == "__main__":
    main()
