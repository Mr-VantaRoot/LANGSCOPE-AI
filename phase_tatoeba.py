"""Archived ten-language Tatoeba experiment; report-only, reruns disabled.

This historical experiment mixed Tatoeba with FLORES+ for Oromo and already
scored its test examples. It remains for provenance and archived-report output;
it must not be used to make or report current project results.
"""

import csv
import argparse
import bz2
import json
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import joblib
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
TATOEBA_DIR = PROJECT_DIR / "data" / "tatoeba"
FLORES_DEV_DIR = PROJECT_DIR / "data" / "flores_plus" / "dev"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "tatoeba"
SEED = 42
TEST_PER_LANGUAGE = 10
TRAIN_FRACTION = 0.80
REGULARIZATION_VALUES = [0.1, 1.0, 10.0]

LANGUAGES = [
    ("Amharic", "amh"),
    ("Afaan Oromo (West Central Oromo)", "gaz"),
    ("Mandarin Chinese", "cmn"),
    ("Japanese", "jpn"),
    ("Finnish", "fin"),
    ("Georgian", "kat"),
    ("Thai", "tha"),
    ("Swahili", "swh"),
    ("Basque", "eus"),
    ("Turkish", "tur"),
]
LABELS = [label for label, _ in LANGUAGES]
LANGDETECT_SUPPORTED_LABELS = {
    "Mandarin Chinese", "Japanese", "Finnish", "Thai", "Swahili", "Turkish"
}

# langdetect supports six of this project's labels. Chinese has two profiles,
# both counted as Mandarin. Amharic, Oromo, Georgian, and Basque are unsupported.
LANGDETECT_TO_LABEL = {
    "zh-cn": "Mandarin Chinese",
    "zh-tw": "Mandarin Chinese",
    "ja": "Japanese",
    "fi": "Finnish",
    "th": "Thai",
    "sw": "Swahili",
    "tr": "Turkish",
}


def read_tatoeba_file(code):
    """Read and validate Tatoeba's ID, ISO 639-3 code, and sentence fields."""
    path = TATOEBA_DIR / f"{code}_sentences.tsv.bz2"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Tatoeba archive: {path}")

    rows = []
    seen_ids = set()
    with bz2.open(path, "rt", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            fields = line.rstrip("\r\n").split("\t", 2)
            if len(fields) != 3:
                raise ValueError(f"Expected three fields in {path}:{line_number}")
            sentence_id, row_code, text = fields
            if row_code != code:
                raise ValueError(
                    f"Expected language code {code!r}, found {row_code!r} "
                    f"in {path}:{line_number}"
                )
            if not sentence_id or not text.strip():
                raise ValueError(f"Empty sentence ID or text in {path}:{line_number}")
            if sentence_id in seen_ids:
                raise ValueError(f"Duplicate Tatoeba sentence ID {sentence_id} in {path}")
            seen_ids.add(sentence_id)
            rows.append(
                {
                    "id": sentence_id,
                    "text": text,
                    "label": dict((code_value, name) for name, code_value in LANGUAGES)[code],
                    "source": "Tatoeba",
                }
            )
    return rows


def read_flores_oromo_dev():
    """Read authentic West Central Oromo text from the existing FLORES+ dev export."""
    path = FLORES_DEV_DIR / "gaz_Latn.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FLORES+ Oromo dev export: {path}")

    rows = []
    seen_ids = set()
    with path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            item = json.loads(line)
            sentence_id = str(item.get("id", ""))
            text = str(item.get("text", ""))
            if not sentence_id or not text.strip():
                raise ValueError(f"Empty sentence ID or text in {path}:{line_number}")
            if sentence_id in seen_ids:
                raise ValueError(f"Duplicate FLORES+ sentence ID {sentence_id} in {path}")
            seen_ids.add(sentence_id)
            rows.append(
                {
                    "id": sentence_id,
                    "text": text,
                    "label": "Afaan Oromo (West Central Oromo)",
                    "source": "FLORES+ dev",
                }
            )
    return rows


def shuffle_rows(rows, rng):
    """Return a shuffled copy so callers do not mutate source records."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled


def split_candidates(rows, rng):
    """Make a deterministic 80/20 train/validation split for one language."""
    shuffled = shuffle_rows(rows, rng)
    train_count = int(len(shuffled) * TRAIN_FRACTION + 0.5)
    return shuffled[:train_count], shuffled[train_count:]


def build_splits():
    """Build equal class splits and keep the ten Tatoeba samples as the test set."""
    rng = random.Random(SEED)
    tatoeba_by_label = {}
    source_counts = {}

    # Read every requested Tatoeba export before making any model predictions.
    for label, code in LANGUAGES:
        rows = read_tatoeba_file(code)
        tatoeba_by_label[label] = rows
        source_counts[label] = {"Tatoeba_total": len(rows)}

    test_by_label = {}
    for label, _ in LANGUAGES:
        rows = shuffle_rows(tatoeba_by_label[label], rng)
        if len(rows) < TEST_PER_LANGUAGE:
            raise ValueError(
                f"{label} has {len(rows)} Tatoeba rows; need at least "
                f"{TEST_PER_LANGUAGE} for the equal test set"
            )
        test_by_label[label] = rows[:TEST_PER_LANGUAGE]
        tatoeba_by_label[label] = rows[TEST_PER_LANGUAGE:]

    train_candidates = {}
    validation_candidates = {}
    for label, code in LANGUAGES:
        if code == "gaz":
            oromo_rows = read_flores_oromo_dev()
            source_counts[label]["FLORES+_dev_total"] = len(oromo_rows)
            train_candidates[label], validation_candidates[label] = split_candidates(
                oromo_rows, rng
            )
        else:
            source_counts[label]["Tatoeba_after_test_reservation"] = len(
                tatoeba_by_label[label]
            )
            train_candidates[label], validation_candidates[label] = split_candidates(
                tatoeba_by_label[label], rng
            )

    # Remove repeated sentence text inside a class and prevent any exact test
    # sentence from leaking into training or validation, even across sources.
    test_texts = {
        text
        for rows in test_by_label.values()
        for text in (row["text"] for row in rows)
    }
    duplicate_counts = {"train_removed": 0, "validation_removed": 0}
    for label, _ in LANGUAGES:
        seen = set()
        kept_train = []
        for row in train_candidates[label]:
            if row["text"] in test_texts or row["text"] in seen:
                duplicate_counts["train_removed"] += 1
                continue
            seen.add(row["text"])
            kept_train.append(row)
        train_candidates[label] = kept_train

        seen = {row["text"] for row in kept_train}
        kept_validation = []
        for row in validation_candidates[label]:
            if row["text"] in test_texts or row["text"] in seen:
                duplicate_counts["validation_removed"] += 1
                continue
            seen.add(row["text"])
            kept_validation.append(row)
        validation_candidates[label] = kept_validation

    train_count = min(len(train_candidates[label]) for label in LABELS)
    validation_count = min(len(validation_candidates[label]) for label in LABELS)
    if train_count < 2 or validation_count < 2:
        raise ValueError("Too few unique sentences remain to create balanced splits")

    train_rows = []
    validation_rows = []
    test_rows = []
    for label, _ in LANGUAGES:
        train_rows.extend(train_candidates[label][:train_count])
        validation_rows.extend(validation_candidates[label][:validation_count])
        test_rows.extend(test_by_label[label])

    # Check for accidental exact text overlap across the final partitions.
    train_texts = {row["text"] for row in train_rows}
    validation_texts = {row["text"] for row in validation_rows}
    final_overlap_counts = {
        "train_validation": len(train_texts & validation_texts),
        "train_test": len(train_texts & test_texts),
        "validation_test": len(validation_texts & test_texts),
    }
    if any(final_overlap_counts.values()):
        raise ValueError(f"Exact sentence leakage across splits: {final_overlap_counts}")

    counts = {
        "before_balancing": {
            "train_per_language": {
                label: len(train_candidates[label]) for label in LABELS
            },
            "validation_per_language": {
                label: len(validation_candidates[label]) for label in LABELS
            },
            "test_per_language": {
                label: len(test_by_label[label]) for label in LABELS
            },
        },
        "after_balancing": {
            "train_per_language": {label: train_count for label in LABELS},
            "validation_per_language": {
                label: validation_count for label in LABELS
            },
            "test_per_language": {
                label: TEST_PER_LANGUAGE for label in LABELS
            },
        },
        "source_sentence_counts": source_counts,
        "removed_exact_duplicates": duplicate_counts,
        "exact_duplicate_overlap_across_final_splits": final_overlap_counts,
        "seed": SEED,
        "test_source": "Tatoeba per-language exports",
        "oromo_train_validation_source": "FLORES+ gaz_Latn dev",
    }
    return train_rows, validation_rows, test_rows, counts


def detect_with_langdetect(text):
    """Return a mapped langdetect label, raw code, or a clear no-detection marker."""
    try:
        raw_code = detect(text)
    except LangDetectException:
        return "no detection", None
    return LANGDETECT_TO_LABEL.get(raw_code, "unsupported prediction"), raw_code


def print_report(metadata):
    """Print the saved scores without rerunning the held-out test."""
    test = metadata["test"]
    detector = metadata["langdetect"]
    print(
        f"Model test accuracy: {test['accuracy']:.6f} "
        f"(n={test['examples']})"
    )
    print(
        "Majority-class baseline accuracy: "
        f"{test['majority_class_baseline_accuracy']:.6f}"
    )
    print("Per-language accuracy: model / langdetect")
    for label in metadata["languages"]:
        scores = test["per_language"][label]
        baseline = scores["langdetect_accuracy"]
        baseline_text = f"{baseline:.6f}" if isinstance(baseline, (int, float)) else baseline
        print(
            f"  {label}: {scores['model_accuracy']:.6f} / "
            f"{baseline_text} (n={scores['support']})"
        )
    print(
        f"langdetect {detector['version']} accuracy: "
        f"{detector['accuracy_on_supported_true_language_examples']:.6f} "
        f"(n={detector['supported_examples']} supported-language examples)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print saved results without rerunning the test set",
    )
    args = parser.parse_args()
    metrics_path = OUTPUT_DIR / "metrics.json"
    if args.report_only:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"No saved Tatoeba metrics found: {metrics_path}")
        print_report(json.loads(metrics_path.read_text(encoding="utf-8")))
        return

    raise SystemExit(
        "This ten-language experiment is archived and its test examples have "
        "already been scored. Rerunning is disabled to prevent test-set reuse. "
        "Use phase_tatoeba_nine_train.py for the current nine-language run; "
        "its saved metrics guard against reevaluation."
    )

    DetectorFactory.seed = SEED
    train_rows, validation_rows, test_rows, split_counts = build_splits()
    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]
    validation_labels = [row["label"] for row in validation_rows]
    test_texts = [row["text"] for row in test_rows]
    test_labels = [row["label"] for row in test_rows]

    print("Balanced sentence counts per language:")
    for label in LABELS:
        print(
            f"  {label}: train={split_counts['after_balancing']['train_per_language'][label]}, "
            f"validation={split_counts['after_balancing']['validation_per_language'][label]}, "
            f"test={split_counts['after_balancing']['test_per_language'][label]}"
        )
    print(
        "Validation tuning uses Logistic Regression C values "
        f"{REGULARIZATION_VALUES}; TF-IDF uses char n-grams (1, 3)."
    )

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    train_features = vectorizer.fit_transform(train_texts)
    validation_features = vectorizer.transform(validation_texts)

    best_score = None
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
        macro_f1 = f1_score(
            validation_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )
        accuracy = accuracy_score(validation_labels, predictions)
        validation_grid.append(
            {
                "C": c_value,
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
            }
        )
        print(
            f"  C={c_value}: validation accuracy={accuracy:.6f}, "
            f"macro-F1={macro_f1:.6f}"
        )
        score = (macro_f1, accuracy)
        if best_score is None or score > best_score:
            best_score = score
            best_classifier = classifier

    # The held-out Tatoeba test rows are first evaluated only after validation
    # selected C. They are passed unchanged to both classifiers below.
    test_features = vectorizer.transform(test_texts)
    model_predictions = best_classifier.predict(test_features).tolist()
    model_scores = best_classifier.predict_proba(test_features)
    model_accuracy = float(accuracy_score(test_labels, model_predictions))
    matrix = confusion_matrix(test_labels, model_predictions, labels=LABELS)
    majority_baseline_accuracy = max(Counter(test_labels).values()) / len(test_labels)

    detector_predictions = []
    detector_codes = []
    for text in test_texts:
        label, raw_code = detect_with_langdetect(text)
        detector_predictions.append(label)
        detector_codes.append(raw_code or "")

    detector_evaluable = [
        index
        for index, actual in enumerate(test_labels)
        if actual in LANGDETECT_SUPPORTED_LABELS
    ]
    detector_accuracy = (
        float(
            accuracy_score(
                [test_labels[index] for index in detector_evaluable],
                [detector_predictions[index] for index in detector_evaluable],
            )
        )
        if detector_evaluable
        else None
    )
    per_language = {}
    for label in LABELS:
        subset = [i for i, actual in enumerate(test_labels) if actual == label]
        model_correct = sum(model_predictions[i] == label for i in subset)
        detector_correct = sum(detector_predictions[i] == label for i in subset)
        detector_is_supported = label in LANGDETECT_SUPPORTED_LABELS
        per_language[label] = {
            "support": len(subset),
            "model_correct": model_correct,
            "model_accuracy": model_correct / len(subset),
            "langdetect_support_status": (
                "supported" if detector_is_supported else "unsupported"
            ),
            "langdetect_correct": detector_correct if detector_is_supported else None,
            "langdetect_accuracy": (
                detector_correct / len(subset)
                if detector_is_supported
                else "unsupported"
            ),
            "langdetect_raw_codes": dict(
                Counter(detector_codes[i] for i in subset)
            ),
        }

    # Prefixes are derived by truncating authentic test sentences; no new
    # language examples are authored or translated.
    short_results = {}
    for length in (5, 10, 20):
        short_texts = [text[:length] for text in test_texts]
        short_predictions = best_classifier.predict(
            vectorizer.transform(short_texts)
        ).tolist()
        short_results[str(length)] = {
            "description": f"first {length} Unicode code points of each Tatoeba test sentence",
            "examples": len(short_texts),
            "accuracy": float(accuracy_score(test_labels, short_predictions)),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, OUTPUT_DIR / "vectorizer.joblib")
    joblib.dump(best_classifier, OUTPUT_DIR / "classifier.joblib")

    with (OUTPUT_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["true_label/predicted_label", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row.tolist()])

    with (OUTPUT_DIR / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as predictions_file:
        writer = csv.writer(predictions_file)
        writer.writerow(
            [
                "sentence_id",
                "source",
                "true_language",
                "text",
                "model_prediction",
                "model_score",
                "langdetect_prediction",
                "langdetect_code",
            ]
        )
        for index, row in enumerate(test_rows):
            class_index = list(best_classifier.classes_).index(model_predictions[index])
            writer.writerow(
                [
                    row["id"],
                    row["source"],
                    row["label"],
                    row["text"],
                    model_predictions[index],
                    float(model_scores[index, class_index]),
                    detector_predictions[index],
                    detector_codes[index],
                ]
            )

    # Preserve real mistakes only. Fewer than five errors are reported honestly.
    failures = []
    for row, prediction in zip(test_rows, model_predictions):
        if prediction != row["label"]:
            if len(row["text"]) < 20:
                explanation = (
                    "This is a short sentence with limited character context; "
                    "that may leave fewer distinctive n-grams for the model."
                )
            else:
                explanation = (
                    "The Oromo training examples came from FLORES+ while this "
                    "test sentence came from Tatoeba. Source or wording differences "
                    "may contribute to this error; this is a hypothesis, not a proven cause."
                )
            failures.append(
                {
                    "sentence_id": row["id"],
                    "source": row["source"],
                    "text": row["text"],
                    "true_language": row["label"],
                    "predicted_language": prediction,
                    "likely_reason": explanation,
                }
            )
    (OUTPUT_DIR / "failure_cases.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "dataset": "Tatoeba per-language sentences with FLORES+ dev supplement for Oromo train/validation",
        "languages": LABELS,
        "seed": SEED,
        "split_counts": split_counts,
        "model": {
            "vectorizer": "TfidfVectorizer(analyzer='char', ngram_range=(1, 3))",
            "classifier": "LogisticRegression",
            "selected_C": best_classifier.C,
            "validation_grid": validation_grid,
            "selection_metric": "validation macro-F1, then validation accuracy",
        },
        "test": {
            "source": "Tatoeba",
            "examples": len(test_rows),
            "examples_per_language": TEST_PER_LANGUAGE,
            "accuracy": model_accuracy,
            "majority_class_baseline_accuracy": majority_baseline_accuracy,
            "per_language": per_language,
            "confusion_matrix_labels": LABELS,
            "test_used_once": True,
            "limitations": [
                "Ten Tatoeba test sentences per language give noisy per-language estimates.",
                "Oromo training and validation use FLORES+ because Tatoeba has only ten Oromo sentences.",
                "The Oromo source differs from the other training sources, so source and language effects may be confounded.",
                "Mixed-language text was not evaluated because no authentic labeled mixed-language test set was included.",
            ],
        },
        "langdetect": {
            "version": version("langdetect"),
            "accuracy_on_supported_true_language_examples": detector_accuracy,
            "supported_examples": len(detector_evaluable),
            "unsupported_languages": [
                label for label in LABELS if label not in LANGDETECT_SUPPORTED_LABELS
            ],
            "out_of_target_predictions_on_supported_examples": sum(
                detector_predictions[index] not in LABELS
                for index in detector_evaluable
            ),
            "per_language": per_language,
        },
        "short_prefix_results": short_results,
        "real_model_failure_count": len(failures),
        "libraries": {"scikit_learn": version("scikit-learn")},
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_report(metadata)
    print(f"Saved Tatoeba experiment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
