"""Choose the app's uncertainty threshold from validation data only."""

import json
import random
from pathlib import Path

import joblib


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "flores_plus" / "dev"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
SEED = 42
VALIDATION_FRACTION = 0.20
MINIMUM_COVERAGE = 0.95
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


def main():
    vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
    classifier = joblib.load(ARTIFACT_DIR / "classifier.joblib")
    rows_by_config = {}
    common_ids = None

    # Read the same dev data and confirm the shared sentence IDs.
    for _, config in LANGUAGES:
        path = DATA_DIR / f"{config}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing dev export: {path}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = {str(row["id"]) for row in rows}
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise ValueError("Sentence IDs do not match across dev files")
        rows_by_config[config] = rows

    # Recreate Phase 2's seeded ID split exactly.
    shuffled_ids = sorted(common_ids)
    random.Random(SEED).shuffle(shuffled_ids)
    validation_count = int(len(shuffled_ids) * VALIDATION_FRACTION + 0.5)
    validation_ids = set(shuffled_ids[:validation_count])

    validation_texts = []
    validation_labels = []
    for label, config in LANGUAGES:
        for row in rows_by_config[config]:
            if str(row["id"]) in validation_ids:
                validation_texts.append(str(row["text"]))
                validation_labels.append(label)

    probabilities = classifier.predict_proba(
        vectorizer.transform(validation_texts)
    )
    confidences = probabilities.max(axis=1)
    predictions = classifier.classes_[probabilities.argmax(axis=1)]

    # Maximize validation accuracy while keeping at least 95% of examples accepted.
    # If several thresholds tie, prefer the highest threshold.
    best_key = None
    best_result = None
    for threshold in sorted(set(float(value) for value in confidences)):
        accepted = confidences >= threshold
        coverage = float(accepted.mean())
        if coverage < MINIMUM_COVERAGE:
            continue
        accepted_correct = sum(
            prediction == truth
            for prediction, truth, keep in zip(
                predictions, validation_labels, accepted
            )
            if keep
        )
        accepted_count = int(accepted.sum())
        selective_accuracy = accepted_correct / accepted_count
        key = (selective_accuracy, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_result = {
                "threshold": threshold,
                "validation_coverage": coverage,
                "validation_selective_accuracy": selective_accuracy,
                "validation_examples": len(validation_labels),
                "accepted_examples": accepted_count,
                "uncertain_examples": len(validation_labels) - accepted_count,
            }

    if best_result is None:
        raise RuntimeError("No threshold met the minimum validation coverage")

    metadata = {
        **best_result,
        "selection_rule": (
            "maximize selective accuracy with at least 95% validation coverage; "
            "break ties by choosing the highest threshold"
        ),
        "seed": SEED,
        "validation_split": "20% of shared FLORES+ dev sentence IDs",
        "test_split_used": False,
    }
    path = ARTIFACT_DIR / "uncertainty_threshold.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected confidence threshold: {best_result['threshold']:.12f}")
    print(f"Validation coverage: {best_result['validation_coverage']:.6f}")
    print(
        "Validation selective accuracy: "
        f"{best_result['validation_selective_accuracy']:.6f}"
    )
    print(
        f"Accepted={best_result['accepted_examples']}; "
        f"uncertain={best_result['uncertain_examples']}; "
        f"validation total={best_result['validation_examples']}"
    )
    print(f"Saved threshold metadata: {path}")


if __name__ == "__main__":
    main()
