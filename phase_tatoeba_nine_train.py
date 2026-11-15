"""Train, calibrate, and evaluate the final balanced nine-language Tatoeba model.

The previous ten-label experiment's test sentence IDs are excluded. A new,
deterministic balanced split is made from the remaining Tatoeba rows. Validation
alone selects Logistic Regression C and the uncertainty threshold. The new test
set is scored once, after those choices are fixed. Existing result files make
the script refuse to run again, preventing accidental test-set reuse.
"""

import bz2
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import joblib
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "tatoeba"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "tatoeba_nine"
ARCHIVED_TESTS = PROJECT_DIR / "artifacts" / "tatoeba" / "test_predictions.csv"
REPLACED_TESTS = (
    PROJECT_DIR
    / "artifacts"
    / "archive"
    / "tatoeba_nine_csv_parser_run"
    / "predictions.csv"
)
LANGUAGES = (
    ("Amharic", "amh"),
    ("Mandarin Chinese", "cmn"),
    ("Japanese", "jpn"),
    ("Finnish", "fin"),
    ("Georgian", "kat"),
    ("Thai", "tha"),
    ("Swahili", "swh"),
    ("Basque", "eus"),
    ("Turkish", "tur"),
)
LABELS = tuple(label for label, _ in LANGUAGES)
LANGDETECT_TO_LABEL = {
    "zh-cn": "Mandarin Chinese",
    "zh-tw": "Mandarin Chinese",
    "ja": "Japanese",
    "fi": "Finnish",
    "th": "Thai",
    "sw": "Swahili",
    "tr": "Turkish",
}
SEED = 20261115
MAX_SENTENCES_PER_LANGUAGE = 2000
SPLIT_FRACTION = 0.15
REGULARIZATION_VALUES = (0.1, 1.0, 10.0)
MINIMUM_VALIDATION_COVERAGE = 0.95


def normalize_text(text):
    """Normalize only for duplicate checks; keep original text for model input."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_previous_test_ids():
    """Read sentence IDs already scored in earlier test runs."""
    if not ARCHIVED_TESTS.is_file():
        raise FileNotFoundError(
            "Archived test prediction manifest is missing; cannot verify that "
            f"old test examples are excluded: {ARCHIVED_TESTS}"
        )
    previous_ids = set()
    for path in (ARCHIVED_TESTS, REPLACED_TESTS):
        if not path.is_file():
            if path == ARCHIVED_TESTS:
                raise FileNotFoundError(f"Archived prediction file is missing: {path}")
            continue
        with path.open(encoding="utf-8", newline="") as file:
            rows = csv.DictReader(file)
            if "sentence_id" not in (rows.fieldnames or ()):
                raise ValueError(f"Prediction file is missing sentence_id: {path}")
            previous_ids.update(row["sentence_id"] for row in rows)
    return previous_ids


def read_language_file(label, code):
    """Read and validate the Tatoeba three-column TSV export for one language."""
    path = DATA_DIR / f"{code}_sentences.tsv.bz2"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Tatoeba archive: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    seen_ids = set()
    with bz2.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.rstrip("\r\n").split("\t", 2)
            if len(fields) != 3:
                raise ValueError(f"Expected ID, language code, and text at {path}:{line_number}")
            sentence_id, row_code, text = fields
            if row_code != code:
                raise ValueError(
                    f"Expected code {code!r}, got {row_code!r} at {path}:{line_number}"
                )
            if not sentence_id or not text.strip():
                raise ValueError(f"Empty ID or sentence at {path}:{line_number}")
            if sentence_id in seen_ids:
                raise ValueError(f"Repeated sentence ID {sentence_id!r} in {path}")
            seen_ids.add(sentence_id)
            records.append(
                {
                    "id": sentence_id,
                    "text": text,
                    "label": label,
                    "normalized_text": normalize_text(text),
                }
            )
    return records, digest


def create_balanced_splits(previous_test_ids):
    """Create balanced train/validation/test partitions with no text overlap."""
    by_label = {}
    file_hashes = {}
    source_counts = {}
    removed_old_test = {}
    removed_within_class_duplicates = {}
    all_sentence_ids = set()
    for label, code in LANGUAGES:
        records, file_hashes[code] = read_language_file(label, code)
        duplicate_ids_across_files = all_sentence_ids & {row["id"] for row in records}
        if duplicate_ids_across_files:
            raise ValueError(
                "Sentence IDs must be unique across language exports; found "
                f"{len(duplicate_ids_across_files)} repeated IDs."
            )
        all_sentence_ids.update(row["id"] for row in records)
        source_counts[label] = len(records)
        removed_old_test[label] = sum(
            row["id"] in previous_test_ids for row in records
        )
        records = [row for row in records if row["id"] not in previous_test_ids]

        unique_rows = []
        seen_texts = set()
        for row in records:
            if row["normalized_text"] in seen_texts:
                continue
            seen_texts.add(row["normalized_text"])
            unique_rows.append(row)
        removed_within_class_duplicates[label] = len(records) - len(unique_rows)
        by_label[label] = unique_rows

    # Exact text appearing under different language labels is ambiguous ground
    # truth for this task. Exclude every member of each such cross-label group.
    text_labels = defaultdict(set)
    for label, rows in by_label.items():
        for row in rows:
            text_labels[row["normalized_text"]].add(label)
    ambiguous_texts = {
        text for text, labels in text_labels.items() if len(labels) > 1
    }
    removed_cross_label_ambiguous = {
        label: sum(row["normalized_text"] in ambiguous_texts for row in rows)
        for label, rows in by_label.items()
    }
    for label in LABELS:
        by_label[label] = [
            row for row in by_label[label]
            if row["normalized_text"] not in ambiguous_texts
        ]

    total_per_language = min(
        MAX_SENTENCES_PER_LANGUAGE,
        min(len(by_label[label]) for label in LABELS),
    )
    if total_per_language < 30:
        raise ValueError(
            "Too few unique, non-previously-tested sentences remain per label: "
            f"{total_per_language}"
        )

    test_count = int(round(total_per_language * SPLIT_FRACTION))
    validation_count = int(round(total_per_language * SPLIT_FRACTION))
    train_count = total_per_language - test_count - validation_count
    splits = {"train": [], "validation": [], "test": []}
    for label in LABELS:
        # Independent deterministic shuffle per class prevents file order from
        # controlling which examples enter a partition.
        label_seed = f"{SEED}:{label}"
        shuffled = list(by_label[label])
        random.Random(label_seed).shuffle(shuffled)
        selected = shuffled[:total_per_language]
        splits["train"].extend(selected[:train_count])
        splits["validation"].extend(
            selected[train_count : train_count + validation_count]
        )
        splits["test"].extend(selected[train_count + validation_count :])

    partition_texts = {
        split_name: {row["normalized_text"] for row in rows}
        for split_name, rows in splits.items()
    }
    overlaps = {
        "train_validation": len(partition_texts["train"] & partition_texts["validation"]),
        "train_test": len(partition_texts["train"] & partition_texts["test"]),
        "validation_test": len(partition_texts["validation"] & partition_texts["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Text leakage across partitions: {overlaps}")

    split_counts = {
        split_name: {label: sum(row["label"] == label for row in rows) for label in LABELS}
        for split_name, rows in splits.items()
    }
    metadata = {
        "seed": SEED,
        "languages": list(LABELS),
        "source_rows_per_language": source_counts,
        "previously_evaluated_test_rows_excluded": removed_old_test,
        "within_language_duplicate_texts_removed": removed_within_class_duplicates,
        "cross_language_ambiguous_text_groups_removed": len(ambiguous_texts),
        "cross_language_ambiguous_rows_removed": removed_cross_label_ambiguous,
        "balanced_unique_sentences_per_language": total_per_language,
        "split_counts_per_language": split_counts,
        "normalized_text_overlap_between_splits": overlaps,
        "previous_test_ids_excluded": len(previous_test_ids),
        "archive_sha256": file_hashes,
        "split_policy": (
            "After excluding all IDs in the archived evaluated test predictions "
            "and all exact normalized cross-language text collisions, shuffle "
            "each language deterministically, take equal class counts, and split "
            "into disjoint train, validation, and test partitions."
        ),
        "max_sentences_per_language": MAX_SENTENCES_PER_LANGUAGE,
    }
    return splits, metadata


def tune_and_calibrate(train_rows, validation_rows):
    """Choose regularization and uncertainty threshold using validation only."""
    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]
    validation_labels = [row["label"] for row in validation_rows]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    train_matrix = vectorizer.fit_transform(train_texts)
    validation_matrix = vectorizer.transform(validation_texts)

    grid = []
    selected = None
    selected_key = None
    for c_value in REGULARIZATION_VALUES:
        model = LogisticRegression(C=c_value, max_iter=2000, random_state=SEED)
        model.fit(train_matrix, train_labels)
        predictions = model.predict(validation_matrix)
        accuracy = float(accuracy_score(validation_labels, predictions))
        macro_f1 = float(
            f1_score(
                validation_labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )
        grid.append({"C": c_value, "accuracy": accuracy, "macro_f1": macro_f1})
        print(f"Validation C={c_value}: accuracy={accuracy:.6f}, macro-F1={macro_f1:.6f}")
        key = (macro_f1, accuracy)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected = model

    validation_scores = selected.predict_proba(validation_matrix)
    confidences = validation_scores.max(axis=1)
    validation_predictions = selected.classes_[validation_scores.argmax(axis=1)]
    best_threshold = None
    best_key = None
    calibration = None
    for threshold in sorted(set(float(score) for score in confidences)):
        accepted = confidences >= threshold
        coverage = float(accepted.mean())
        if coverage < MINIMUM_VALIDATION_COVERAGE:
            continue
        accepted_correct = sum(
            predicted == actual
            for predicted, actual, keep in zip(
                validation_predictions, validation_labels, accepted
            )
            if keep
        )
        selective_accuracy = accepted_correct / int(accepted.sum())
        key = (selective_accuracy, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            calibration = {
                "threshold": threshold,
                "validation_coverage": coverage,
                "validation_selective_accuracy": selective_accuracy,
                "validation_examples": len(validation_labels),
                "minimum_coverage": MINIMUM_VALIDATION_COVERAGE,
                "test_split_used": False,
            }
    if best_threshold is None:
        raise RuntimeError("No validation threshold met the required coverage.")
    return vectorizer, selected, grid, calibration


def detect_with_langdetect(text):
    """Map langdetect output to this project or mark it unsupported."""
    try:
        raw_code = detect(text)
    except LangDetectException:
        return "no detection", ""
    return LANGDETECT_TO_LABEL.get(raw_code, "unsupported prediction"), raw_code


def score_once(vectorizer, model, test_rows):
    """Score the frozen model and langdetect once on identical held-out texts."""
    test_texts = [row["text"] for row in test_rows]
    truth = [row["label"] for row in test_rows]
    features = vectorizer.transform(test_texts)
    predictions = model.predict(features).tolist()
    score_matrix = model.predict_proba(features)
    top_scores = score_matrix.max(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(truth, predictions, labels=LABELS)
    per_language = {}
    for index, label in enumerate(LABELS):
        per_language[label] = {
            "support": int(support[index]),
            "accuracy": float(
                sum(p == y for p, y in zip(predictions, truth) if y == label)
                / support[index]
            ),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }

    DetectorFactory.seed = SEED
    detector_predictions = []
    detector_codes = []
    for text in test_texts:
        predicted, code = detect_with_langdetect(text)
        detector_predictions.append(predicted)
        detector_codes.append(code)
    evaluable = [
        index for index, label in enumerate(truth)
        if label in LANGDETECT_TO_LABEL.values()
    ]
    supported_truth = [truth[index] for index in evaluable]
    supported_predictions = [detector_predictions[index] for index in evaluable]
    detector_accuracy = (
        float(accuracy_score(supported_truth, supported_predictions))
        if evaluable
        else None
    )
    supported_labels = sorted(set(LANGDETECT_TO_LABEL.values()))
    d_precision, d_recall, d_f1, d_support = precision_recall_fscore_support(
        supported_truth,
        supported_predictions,
        labels=supported_labels,
        zero_division=0,
    ) if supported_truth else ([], [], [], [])
    detector_per_language = {}
    for index, label in enumerate(supported_labels):
        detector_per_language[label] = {
            "support": int(d_support[index]),
            "accuracy": float(d_recall[index]),
            "precision": float(d_precision[index]),
            "recall": float(d_recall[index]),
            "f1": float(d_f1[index]),
        }

    lengths = [len(row["text"]) for row in test_rows]
    short_summary = {}
    for name, condition in (
        ("1-10 characters", lambda length: length <= 10),
        ("11-20 characters", lambda length: 11 <= length <= 20),
        ("over 20 characters", lambda length: length > 20),
    ):
        indices = [i for i, length in enumerate(lengths) if condition(length)]
        short_summary[name] = {
            "examples": len(indices),
            "accuracy": (
                float(accuracy_score([truth[i] for i in indices], [predictions[i] for i in indices]))
                if indices
                else None
            ),
        }

    failures = []
    for index, row in enumerate(test_rows):
        if predictions[index] != truth[index]:
            failures.append(
                {
                    "sentence_id": row["id"],
                    "text": row["text"],
                    "true_language": truth[index],
                    "predicted_language": predictions[index],
                    "top_model_score": float(top_scores[index]),
                    "text_length_characters": lengths[index],
                }
            )

    model_accuracy = float(accuracy_score(truth, predictions))
    majority_accuracy = max(Counter(truth).values()) / len(truth)
    result = {
        "test_examples": len(test_rows),
        "accuracy": model_accuracy,
        "macro_precision": float(sum(precision) / len(precision)),
        "macro_recall": float(sum(recall) / len(recall)),
        "macro_f1": float(sum(f1) / len(f1)),
        "majority_class_baseline_accuracy": float(majority_accuracy),
        "per_language": per_language,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": matrix.tolist(),
        "langdetect": {
            "version": version("langdetect"),
            "supported_languages": supported_labels,
            "unsupported_languages": [
                label for label in LABELS if label not in supported_labels
            ],
            "scored_examples": len(evaluable),
            "coverage": len(evaluable) / len(test_rows),
            "accuracy_on_supported_examples": detector_accuracy,
            "per_language": detector_per_language,
        },
        "text_length_groups": short_summary,
        "failure_count": len(failures),
    }

    predictions_rows = []
    for index, row in enumerate(test_rows):
        predictions_rows.append(
            {
                "sentence_id": row["id"],
                "true_language": truth[index],
                "text": row["text"],
                "text_length_characters": lengths[index],
                "model_prediction": predictions[index],
                "top_model_score": float(top_scores[index]),
                "langdetect_prediction": detector_predictions[index],
                "langdetect_code": detector_codes[index],
            }
        )
    return result, failures, predictions_rows


def main():
    metrics_path = OUTPUT_DIR / "metrics.json"
    result_files = (
        metrics_path,
        OUTPUT_DIR / "predictions.csv",
        OUTPUT_DIR / "split_manifest.json",
        OUTPUT_DIR / "confusion_matrix.csv",
    )
    if any(path.exists() for path in result_files):
        raise FileExistsError(
            "Refusing to rerun or overwrite the one-time test evaluation. "
            f"Existing results: {[str(path) for path in result_files if path.exists()]}"
        )
    if not ARCHIVED_TESTS.is_file():
        raise FileNotFoundError(
            "Cannot exclude prior evaluated test rows; archived prediction manifest "
            f"is missing: {ARCHIVED_TESTS}"
        )

    previous_test_ids = read_previous_test_ids()
    splits, split_metadata = create_balanced_splits(previous_test_ids)
    print("Balanced data counts per language:")
    for split_name, counts in split_metadata["split_counts_per_language"].items():
        print(f"  {split_name}: {set(counts.values()).pop()} each")
    print(
        "Previously evaluated test rows excluded from source: "
        f"{split_metadata['previous_test_ids_excluded']} IDs."
    )
    print(
        "Exact same normalized text across different labels removed: "
        f"{split_metadata['cross_language_ambiguous_text_groups_removed']} groups."
    )
    print("Tuning C and confidence threshold with validation only.")
    vectorizer, model, validation_grid, calibration = tune_and_calibrate(
        splits["train"], splits["validation"]
    )

    # This is the sole evaluation of this final model on its held-out test split.
    test_metrics, failures, prediction_rows = score_once(
        vectorizer, model, splits["test"]
    )

    # All calculations are complete before artifacts are written; refuse a
    # partial run if output already existed at startup.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, OUTPUT_DIR / "vectorizer.joblib")
    joblib.dump(model, OUTPUT_DIR / "classifier.joblib")
    (OUTPUT_DIR / "uncertainty_threshold.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    train_counts = split_metadata["split_counts_per_language"]["train"]
    validation_counts = split_metadata["split_counts_per_language"]["validation"]
    test_counts = split_metadata["split_counts_per_language"]["test"]
    training_metadata = {
        "dataset": "Tatoeba per-language sentence exports",
        "languages": list(LABELS),
        "seed": SEED,
        "training_source": "Tatoeba only",
        "training_examples": sum(train_counts.values()),
        "validation_examples": sum(validation_counts.values()),
        "test_examples": sum(test_counts.values()),
        "train_per_language": train_counts,
        "validation_per_language": validation_counts,
        "test_per_language": test_counts,
        "previous_test_set_reused": False,
        "previously_scored_test_ids_excluded": True,
        "fresh_test_set_required": False,
        "final_test_evaluation_completed": True,
        "model": {
            "vectorizer": "TfidfVectorizer(analyzer='char', ngram_range=(1, 3))",
            "classifier": "LogisticRegression",
            "selected_C": model.C,
            "validation_grid": validation_grid,
            "selection_metric": "validation macro-F1, then validation accuracy",
        },
        "threshold": calibration,
        "data": split_metadata,
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "split_manifest.json").write_text(
        json.dumps(
            {
                name: [
                    {"sentence_id": row["id"], "language": row["label"]}
                    for row in rows
                ]
                for name, rows in splits.items()
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_rows = [
        next(row for row in splits["train"] if row["label"] == label)
        for label in LABELS
    ]
    (OUTPUT_DIR / "training_examples.json").write_text(
        json.dumps(
            [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "label": row["label"],
                    "source": "Tatoeba",
                }
                for row in sample_rows
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "Tatoeba per-language sentence exports",
                "languages": list(LABELS),
                "training": {
                    "model": "LogisticRegression",
                    "vectorizer": "TfidfVectorizer(analyzer='char', ngram_range=(1, 3))",
                    "selected_C": model.C,
                    "validation_grid": validation_grid,
                    "selection_metric": "validation macro-F1, then validation accuracy",
                },
                "uncertainty_calibration": calibration,
                "data": split_metadata,
                "libraries": {
                    package: version(package)
                    for package in ("scikit-learn", "langdetect", "numpy", "scipy")
                },
                "test": test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "failure_cases.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (OUTPUT_DIR / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true_label/predicted_label", *LABELS])
        for label, row in zip(LABELS, test_metrics["confusion_matrix"]):
            writer.writerow([label, *row])

    print(
        f"Final held-out test accuracy: {test_metrics['accuracy']:.6f} "
        f"(n={test_metrics['test_examples']})"
    )
    print(
        "langdetect accuracy on its supported languages only: "
        f"{test_metrics['langdetect']['accuracy_on_supported_examples']:.6f} "
        f"(n={test_metrics['langdetect']['scored_examples']}; "
        f"coverage={test_metrics['langdetect']['coverage']:.6f})"
    )
    print(f"Final test misclassifications saved: {len(failures)}")
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
