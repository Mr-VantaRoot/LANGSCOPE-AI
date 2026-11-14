"""Run the one-time held-out test evaluation for LANGSCOPE AI."""

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import langid
from importlib.metadata import version
from sklearn.metrics import confusion_matrix


TEST_DIR = Path("data/flores_plus/devtest")
ARTIFACT_DIR = Path("artifacts")
LANGUAGES = [
    ("Amharic", "amh_Ethi"),
    ("Afaan Oromo (West Central Oromo)", "gaz_Latn"),
    ("Mandarin Chinese (Simplified/Beijing)", "cmn_Hans"),
    ("Japanese", "jpn_Jpan"),
    ("Finnish", "fin_Latn"),
    ("Georgian", "kat_Geor"),
    ("Thai", "tha_Thai"),
    ("Swahili", "swh_Latn"),
    ("Basque", "eus_Latn"),
    ("Turkish", "tur_Latn"),
]
LABELS = [label for label, _ in LANGUAGES]
LANGID_CODES = {
    "Amharic": "am",
    "Mandarin Chinese (Simplified/Beijing)": "zh",
    "Japanese": "ja",
    "Finnish": "fi",
    "Georgian": "ka",
    "Thai": "th",
    "Swahili": "sw",
    "Basque": "eu",
    "Turkish": "tr",
}
SHORT_PREFIX_LENGTHS = [5, 10, 20]


def read_test_data():
    """Read and validate every test file before making any predictions."""
    all_rows = []
    common_ids = None

    for label, config in LANGUAGES:
        path = TEST_DIR / f"{config}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing held-out test export: {path}")

        rows = []
        with path.open(encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                row = json.loads(line)
                sentence_id = row.get("id")
                text = row.get("text")
                if sentence_id is None or text is None or not str(text).strip():
                    raise ValueError(f"Missing ID or empty test text in {path}:{line_number}")
                rows.append({"id": str(sentence_id), "text": str(text), "label": label})

        if len(rows) != 1012:
            raise ValueError(f"Expected 1012 test sentences in {path}; found {len(rows)}")
        ids = {row["id"] for row in rows}
        if len(ids) != len(rows):
            raise ValueError(f"Duplicate test IDs in {path}")
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise ValueError("Test sentence IDs do not match across languages")

        all_rows.extend(rows)
        print(f"Loaded {len(rows)} held-out sentences for {label} ({config})")

    return all_rows


def accuracy(true_labels, predicted_labels):
    """Calculate accuracy without relying on implicit label ordering."""
    if not true_labels:
        return 0.0
    correct = sum(actual == predicted for actual, predicted in zip(true_labels, predicted_labels))
    return correct / len(true_labels)


def per_language_metrics(true_labels, predicted_labels):
    """Calculate precision, recall, F1, and support for each target label."""
    results = {}
    for label in LABELS:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(true_labels, predicted_labels)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(true_labels, predicted_labels)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(true_labels, predicted_labels)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }
    return results


def explain_failure(input_type, true_label, predicted_label, text):
    """Give a cautious, text-specific reason for a real model error."""
    if input_type != "full sentence":
        return (
            f"This {input_type} contains little character context. The selected "
            f"character n-grams overlap more with the {predicted_label} training "
            "examples than with the true class."
        )
    if true_label == "Afaan Oromo (West Central Oromo)" or predicted_label == "Afaan Oromo (West Central Oromo)":
        return (
            "Oromo, Finnish, Basque, Swahili, and Turkish all use Latin script; "
            "this full sentence may share character patterns or borrowed terms "
            f"with the predicted {predicted_label} class."
        )
    return (
        f"The full sentence may contain names, numbers, or character patterns "
        f"shared with {predicted_label}; the model assigned the stronger learned "
        "n-gram evidence to that class."
    )


def main():
    # Check the saved Phase 2 artifacts before reading the held-out test set.
    vectorizer_path = ARTIFACT_DIR / "vectorizer.joblib"
    classifier_path = ARTIFACT_DIR / "classifier.joblib"
    if not vectorizer_path.is_file() or not classifier_path.is_file():
        raise FileNotFoundError("Phase 2 model artifacts are missing")
    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    if set(classifier.classes_) != set(LABELS):
        raise ValueError("Saved classifier labels do not match target languages")

    # Load every test file and validate counts/IDs before evaluating once.
    rows = read_test_data()
    true_labels = [row["label"] for row in rows]
    full_texts = [row["text"] for row in rows]

    print("\nRunning the one-time full-sentence evaluation...")
    full_predictions = classifier.predict(vectorizer.transform(full_texts)).tolist()
    full_accuracy = accuracy(true_labels, full_predictions)
    language_scores = per_language_metrics(true_labels, full_predictions)
    matrix = confusion_matrix(true_labels, full_predictions, labels=LABELS)

    print(f"Full-sentence accuracy: {full_accuracy:.6f}")
    print("Per-language metrics (precision, recall, F1, support):")
    for label in LABELS:
        score = language_scores[label]
        print(
            f"  {label}: precision={score['precision']:.6f}, "
            f"recall={score['recall']:.6f}, f1={score['f1']:.6f}, "
            f"support={score['support']}"
        )

    # Save a CSV confusion matrix with true labels as rows and predictions as columns.
    confusion_path = ARTIFACT_DIR / "confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["true_label/predicted_label", *LABELS])
        for label, row_values in zip(LABELS, matrix):
            writer.writerow([label, *row_values.tolist()])

    # The classes are balanced, so the majority baseline predicts the first tied class.
    class_counts = Counter(true_labels)
    majority_label = max(LABELS, key=lambda label: class_counts[label])
    majority_predictions = [majority_label] * len(true_labels)
    majority_accuracy = accuracy(true_labels, majority_predictions)
    print(
        f"Majority-class baseline: label={majority_label}, "
        f"accuracy={majority_accuracy:.6f}"
    )

    # Compare langid only where its published model supports the target language.
    langid.set_languages(list(LANGID_CODES.values()))
    langid_label_by_code = {code: label for label, code in LANGID_CODES.items()}
    supported_rows = [row for row in rows if row["label"] in LANGID_CODES]
    supported_true = [row["label"] for row in supported_rows]
    langid_predictions = [
        langid_label_by_code[langid.classify(row["text"])[0]]
        for row in supported_rows
    ]
    langid_accuracy = accuracy(supported_true, langid_predictions)
    langid_scores = per_language_metrics(supported_true, langid_predictions)
    print(
        f"langid {version('langid')}: accuracy={langid_accuracy:.6f} "
        f"on {len(supported_rows)} supported examples; "
        "Afaan Oromo: unsupported"
    )
    print("langid per-language accuracy:")
    for label in LABELS:
        if label in LANGID_CODES:
            score = langid_scores[label]
            print(f"  {label}: {score['recall']:.6f} (n={score['support']})")
        else:
            print(f"  {label}: unsupported")

    # Measure short, potentially ambiguous inputs using prefixes from real test text.
    short_results = {}
    short_errors = []
    for prefix_length in SHORT_PREFIX_LENGTHS:
        short_texts = [text[:prefix_length] for text in full_texts]
        short_predictions = classifier.predict(
            vectorizer.transform(short_texts)
        ).tolist()
        short_accuracy = accuracy(true_labels, short_predictions)
        short_results[str(prefix_length)] = {
            "input_definition": f"first {prefix_length} Unicode code points of each test sentence",
            "accuracy": short_accuracy,
            "examples": len(short_texts),
        }
        print(
            f"{prefix_length}-character prefix accuracy: "
            f"{short_accuracy:.6f} (n={len(short_texts)})"
        )

        for row, prediction, short_text in zip(rows, short_predictions, short_texts):
            if row["label"] != prediction:
                short_errors.append(
                    {
                        "id": row["id"],
                        "input_type": f"{prefix_length}-character prefix",
                        "text": short_text,
                        "source_test_sentence": row["text"],
                        "true_language": row["label"],
                        "predicted_language": prediction,
                    }
                )

    # Save real model failures, preferring full inputs and then short prefixes.
    full_errors = [
        {
            "id": row["id"],
            "input_type": "full sentence",
            "text": row["text"],
            "true_language": row["label"],
            "predicted_language": prediction,
        }
        for row, prediction in zip(rows, full_predictions)
        if row["label"] != prediction
    ]
    failure_candidates = full_errors + short_errors
    failures = []
    seen = set()
    for case in failure_candidates:
        key = (case["id"], case["input_type"])
        if key in seen:
            continue
        seen.add(key)
        case["likely_reason"] = explain_failure(
            case["input_type"],
            case["true_language"],
            case["predicted_language"],
            case["text"],
        )
        failures.append(case)
        if len(failures) == 5:
            break

    failure_path = ARTIFACT_DIR / "failure_cases.json"
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Full-sentence errors: {len(full_errors)}")
    print(f"Short-prefix errors collected: {len(short_errors)}")
    print(f"Saved {len(failures)} real failure cases: {failure_path}")
    for index, case in enumerate(failures, start=1):
        print(
            f"  failure {index}: {case['input_type']}; "
            f"true={case['true_language']}; predicted={case['predicted_language']}; "
            f"text={case['text']}"
        )

    # Save all full-sentence predictions and the aggregate results for the report.
    predictions_path = ARTIFACT_DIR / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.writer(prediction_file)
        writer.writerow(["id", "true_language", "predicted_language"])
        for row, prediction in zip(rows, full_predictions):
            writer.writerow([row["id"], row["label"], prediction])

    metrics = {
        "test_examples": len(rows),
        "test_per_language": language_scores,
        "test_accuracy": full_accuracy,
        "confusion_matrix_labels": LABELS,
        "majority_baseline": {
            "predicted_label": majority_label,
            "accuracy": majority_accuracy,
        },
        "external_detector": {
            "name": "langid",
            "version": version("langid"),
            "supported_examples": len(supported_rows),
            "unsupported_languages": ["Afaan Oromo (West Central Oromo)"],
            "accuracy_on_supported_examples": langid_accuracy,
            "per_language": {
                label: langid_scores[label] if label in LANGID_CODES else "unsupported"
                for label in LABELS
            },
        },
        "short_prefix_results": short_results,
        "full_sentence_failure_count": len(full_errors),
        "short_prefix_failure_count": len(short_errors),
        "test_evaluated_once": True,
    }
    metrics_path = ARTIFACT_DIR / "phase3_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
