"""Tune and train the LANGSCOPE AI language identifier. The test split is not used."""

import json
import random
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "flores_plus" / "dev"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
SEED = 42
VALIDATION_FRACTION = 0.20

# Keep this order stable so ties in validation scores are reproducible.
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
NGRAM_RANGES = [(1, 3), (2, 4), (3, 5)]
REGULARIZATION_VALUES = [0.1, 1.0, 10.0]


def read_language_rows(config):
    """Read one exported dev file and check its sentence IDs and text."""
    path = DATA_DIR / f"{config}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset export: {path}")

    rows = []
    with path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            row = json.loads(line)
            text = row.get("text")
            sentence_id = row.get("id")
            if sentence_id is None or text is None or not str(text).strip():
                raise ValueError(f"Missing ID or empty text in {path}:{line_number}")
            rows.append({"id": str(sentence_id), "text": str(text)})

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sentence IDs in {path}")
    return rows


def make_train_validation_ids(all_ids):
    """Shuffle shared sentence IDs once, keeping parallel translations aligned."""
    shuffled_ids = sorted(all_ids)
    random.Random(SEED).shuffle(shuffled_ids)
    validation_count = int(len(shuffled_ids) * VALIDATION_FRACTION + 0.5)
    validation_ids = set(shuffled_ids[:validation_count])
    train_ids = set(shuffled_ids[validation_count:])
    return train_ids, validation_ids


def main():
    rows_by_config = {}
    labels_by_config = {config: label for label, config in LANGUAGES}
    common_ids = None

    # Load only the exported dev files. devtest is intentionally untouched.
    for label, config in LANGUAGES:
        rows = read_language_rows(config)
        rows_by_config[config] = rows
        current_ids = {row["id"] for row in rows}
        if common_ids is None:
            common_ids = current_ids
        elif current_ids != common_ids:
            raise ValueError("Sentence IDs do not match across language files")
        print(f"Loaded {len(rows)} dev sentences for {label} ({config})")

    train_ids, validation_ids = make_train_validation_ids(common_ids)
    train_texts, train_labels = [], []
    validation_texts, validation_labels = [], []

    # Apply the same ID split to every language, preserving translation alignment.
    for _, config in LANGUAGES:
        label = labels_by_config[config]
        for row in rows_by_config[config]:
            if row["id"] in train_ids:
                train_texts.append(row["text"])
                train_labels.append(label)
            else:
                validation_texts.append(row["text"])
                validation_labels.append(label)

    print(
        f"\nSplit counts per language: train={len(train_ids)}, "
        f"validation={len(validation_ids)}; seed={SEED}"
    )
    print("Validation tuning: macro-F1 is primary; accuracy breaks ties.")

    results = []
    best_key = None
    best_vectorizer = None
    best_classifier = None
    best_settings = None

    # Evaluate every n-gram and regularization candidate on validation only.
    for ngram_range in NGRAM_RANGES:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=False,
            sublinear_tf=True,
        )
        train_features = vectorizer.fit_transform(train_texts)
        validation_features = vectorizer.transform(validation_texts)

        for c_value in REGULARIZATION_VALUES:
            classifier = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                random_state=SEED,
            )
            classifier.fit(train_features, train_labels)
            predictions = classifier.predict(validation_features)
            validation_accuracy = accuracy_score(validation_labels, predictions)
            validation_macro_f1 = f1_score(
                validation_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
            settings = {"ngram_range": list(ngram_range), "C": c_value}
            results.append(
                {
                    **settings,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_macro_f1,
                }
            )
            print(
                f"ngram_range={ngram_range}, C={c_value}: "
                f"validation_accuracy={validation_accuracy:.6f}, "
                f"validation_macro_f1={validation_macro_f1:.6f}"
            )

            score_key = (validation_macro_f1, validation_accuracy)
            if best_key is None or score_key > best_key:
                best_key = score_key
                best_vectorizer = vectorizer
                best_classifier = classifier
                best_settings = settings

    # Save the selected pipeline, fitted on train only. Validation remains available
    # for confidence-threshold selection in Phase 4.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = ARTIFACT_DIR / "vectorizer.joblib"
    classifier_path = ARTIFACT_DIR / "classifier.joblib"
    metadata_path = ARTIFACT_DIR / "phase2_metadata.json"
    joblib.dump(best_vectorizer, vectorizer_path)
    joblib.dump(best_classifier, classifier_path)

    metadata = {
        "dataset": "openlanguagedata/flores_plus",
        "dataset_configs": [config for _, config in LANGUAGES],
        "label_names": [label for label, _ in LANGUAGES],
        "seed": SEED,
        "train_sentence_ids": len(train_ids),
        "validation_sentence_ids": len(validation_ids),
        "validation_split_method": "one seeded split of shared dev sentence IDs",
        "tuning_metric": "validation macro-F1, accuracy as tie-breaker",
        "selected_settings": best_settings,
        "selected_validation_accuracy": best_key[1],
        "selected_validation_macro_f1": best_key[0],
        "validation_grid": results,
        "scikit_learn_version": sklearn.__version__,
        "final_fit_data": "train split only",
        "test_split_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSelected settings: {best_settings}")
    print(f"Selected validation accuracy: {best_key[1]:.6f}")
    print(f"Selected validation macro-F1: {best_key[0]:.6f}")
    print(f"Saved vectorizer: {vectorizer_path}")
    print(f"Saved classifier: {classifier_path}")
    print(f"Saved tuning metadata: {metadata_path}")
    print("devtest was not loaded or evaluated.")


if __name__ == "__main__":
    main()
