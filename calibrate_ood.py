"""Calibrate and audit an out-of-distribution rejection gate.

This leaves the trained nine-class classifier and its held-out test artifacts
unchanged. The gate threshold is selected from the existing in-distribution
validation split and separate Tatoeba English/French OOD calibration samples.
Disjoint English/French OOD audit samples are reserved for reporting only.
"""

import bz2
import csv
import hashlib
import json
import random
import unicodedata
from pathlib import Path

import joblib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "tatoeba"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "tatoeba_nine"
SEED = 20261115
OOD_CODES = ("eng", "fra")
SAMPLE_PER_PARTITION = 500
MINIMUM_ID_VALIDATION_COVERAGE = 0.95
REGRESSION_PROBE = "How are you?"


def normalize_text(text):
    """Normalize whitespace and Unicode for duplicate checks only."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def read_split_texts(codes, manifest):
    """Recover train and validation text by saved IDs; never read test text."""
    wanted = {}
    for split in ("train", "validation"):
        for row in manifest[split]:
            wanted[row["sentence_id"]] = (split, row["language"])

    recovered = {"train": {}, "validation": {}}
    for code in codes:
        path = DATA_DIR / f"{code}_sentences.tsv.bz2"
        if not path.is_file():
            raise FileNotFoundError(f"Required in-distribution Tatoeba archive missing: {path}")
        with bz2.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                fields = line.rstrip("\r\n").split("\t", 2)
                if len(fields) != 3:
                    raise ValueError(f"Invalid Tatoeba TSV row at {path}:{line_number}")
                sentence_id, row_code, text = fields
                if sentence_id not in wanted:
                    continue
                split, label = wanted[sentence_id]
                if row_code != code or not text.strip():
                    raise ValueError(f"Unexpected Tatoeba row at {path}:{line_number}")
                recovered[split][sentence_id] = (label, text)

    for split in recovered:
        if len(recovered[split]) != sum(1 for row in manifest[split]):
            raise ValueError(f"Could not recover every {split} sentence from the manifest")
    return recovered


def sample_ood_partition(code, known_texts, excluded_test_texts):
    """Uniformly reservoir-sample two ID-disjoint subsets from one OOD export."""
    path = DATA_DIR / f"{code}_sentences.tsv.bz2"
    if not path.is_file():
        raise FileNotFoundError(f"Required OOD Tatoeba archive missing: {path}")

    reservoirs = [[], []]
    counts = [0, 0]
    randomizers = [random.Random(SEED + index + sum(code.encode())) for index in range(2)]
    source_rows = 0
    excluded_known_text = 0
    excluded_probe = 0

    with bz2.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\r\n").split("\t", 2)
            if len(fields) != 3:
                raise ValueError(f"Invalid Tatoeba TSV row at {path}:{line_number}")
            sentence_id, row_code, text = fields
            if row_code != code or not sentence_id or not text.strip():
                raise ValueError(f"Unexpected Tatoeba row at {path}:{line_number}")
            source_rows += 1
            normalized = normalize_text(text)
            if normalized in known_texts:
                excluded_known_text += 1
                continue
            if normalized in excluded_test_texts:
                excluded_probe += 1
                continue
            if not any(character.isalpha() for character in text):
                continue

            # A stable ID hash creates disjoint calibration and audit pools.
            partition = hashlib.sha256(
                f"langs-ood-{SEED}:{code}:{sentence_id}".encode("utf-8")
            ).digest()[0] & 1
            counts[partition] += 1
            record = {"sentence_id": sentence_id, "language_code": code, "text": text}
            reservoir = reservoirs[partition]
            if len(reservoir) < SAMPLE_PER_PARTITION:
                reservoir.append(record)
            else:
                replacement = randomizers[partition].randrange(counts[partition])
                if replacement < SAMPLE_PER_PARTITION:
                    reservoir[replacement] = record

    for partition, name in enumerate(("calibration", "audit")):
        if len(reservoirs[partition]) != SAMPLE_PER_PARTITION:
            raise ValueError(
                f"Only {len(reservoirs[partition])} {code} {name} rows available; "
                f"required {SAMPLE_PER_PARTITION}."
            )
    return reservoirs, {
        "archive": path.name,
        "source_rows": source_rows,
        "eligible_rows_per_partition": counts,
        "sample_rows_per_partition": [len(rows) for rows in reservoirs],
        "excluded_overlap_with_train_or_validation": excluded_known_text,
        "excluded_regression_probe_text": excluded_probe,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def score_texts(vectorizer, classifier, texts):
    """Return raw maximum class score, runner-up score, and all class scores."""
    matrix = vectorizer.transform(texts)
    scores = classifier.predict_proba(matrix)
    order = np.argsort(scores, axis=1)[:, ::-1]
    rows = np.arange(len(texts))
    return scores, order, scores[rows, order[:, 0]], scores[rows, order[:, 1]]


def select_threshold(id_scores, ood_scores):
    """Maximize balanced ID/OOD accuracy subject to 95% ID validation coverage."""
    candidates = np.unique(np.concatenate((id_scores, ood_scores)))
    best = None
    for threshold in candidates:
        id_coverage = float(np.mean(id_scores >= threshold))
        if id_coverage < MINIMUM_ID_VALIDATION_COVERAGE:
            continue
        ood_rejection = float(np.mean(ood_scores < threshold))
        balanced_accuracy = 0.5 * (id_coverage + ood_rejection)
        key = (balanced_accuracy, ood_rejection, id_coverage, -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), id_coverage, ood_rejection)
    if best is None:
        raise RuntimeError("No rejection threshold met the minimum ID validation coverage.")
    _, threshold, id_coverage, ood_rejection = best
    return threshold, id_coverage, ood_rejection


def make_rows(samples, labels, scores, order, top, second, threshold, split):
    output = []
    for index, sample in enumerate(samples):
        all_scores = {
            label: float(scores[index, class_index])
            for class_index, label in enumerate(labels)
        }
        output.append(
            {
                **sample,
                "split": split,
                "top_prediction": labels[int(order[index, 0])],
                "top_score": float(top[index]),
                "second_prediction": labels[int(order[index, 1])],
                "second_score": float(second[index]),
                "accepted_as_supported": bool(top[index] >= threshold),
                "class_scores": all_scores,
            }
        )
    return output


def write_csv(path, rows):
    fieldnames = (
        "sentence_id",
        "language_code",
        "text",
        "split",
        "top_prediction",
        "top_score",
        "second_prediction",
        "second_score",
        "accepted_as_supported",
    )
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main():
    manifest = json.loads((ARTIFACT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    codes = ("amh", "cmn", "jpn", "fin", "kat", "tha", "swh", "eus", "tur")
    recovered = read_split_texts(codes, manifest)
    known_texts = {
        normalize_text(text)
        for split_rows in recovered.values()
        for _, text in split_rows.values()
    }
    exclusions = {normalize_text(REGRESSION_PROBE)}

    source_rows = {}
    source_metadata = {}
    for code in OOD_CODES:
        partitions, metadata = sample_ood_partition(code, known_texts, exclusions)
        source_rows[code] = partitions
        source_metadata[code] = metadata

    vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
    classifier = joblib.load(ARTIFACT_DIR / "classifier.joblib")
    labels = [str(label) for label in classifier.classes_]
    id_validation = [text for _, text in recovered["validation"].values()]
    id_scores, id_order, id_top, id_second = score_texts(vectorizer, classifier, id_validation)

    ood_scores_by_split = {"calibration": [], "audit": []}
    rows_by_split = {"calibration": [], "audit": []}
    for code in OOD_CODES:
        for partition_index, split in enumerate(("calibration", "audit")):
            samples = source_rows[code][partition_index]
            texts = [row["text"] for row in samples]
            scores, order, top, second = score_texts(vectorizer, classifier, texts)
            rows = make_rows(samples, labels, scores, order, top, second, float("inf"), split)
            rows_by_split[split].extend(rows)
            if split == "calibration":
                ood_scores_by_split[split].extend(float(value) for value in top)

    threshold, id_coverage, ood_calibration_rejection = select_threshold(
        id_top, np.asarray(ood_scores_by_split["calibration"])
    )
    for split in ("calibration", "audit"):
        for row in rows_by_split[split]:
            row["accepted_as_supported"] = row["top_score"] >= threshold

    (ARTIFACT_DIR / "ood_rejection.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "score": "maximum Logistic Regression class score",
                "rule": "reject when maximum class score is below this threshold",
                "selection_metric": "maximum balanced accuracy on ID/OOD calibration data",
                "minimum_id_validation_coverage": MINIMUM_ID_VALIDATION_COVERAGE,
                "id_validation_coverage": id_coverage,
                "ood_calibration_rejection_rate": ood_calibration_rejection,
                "balanced_accuracy": 0.5 * (id_coverage + ood_calibration_rejection),
                "ood_languages_for_calibration": list(OOD_CODES),
                "samples_per_ood_language_per_partition": SAMPLE_PER_PARTITION,
                "seed": SEED,
                "uses_final_test_set": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results = {
        "purpose": "separate OOD rejection calibration and audit; not a replacement for the nine-language final evaluation",
        "calibration": {
            "in_distribution_validation_examples": len(id_validation),
            "in_distribution_accepted": int(np.sum(id_top >= threshold)),
            "in_distribution_coverage": id_coverage,
            "out_of_distribution_examples": len(rows_by_split["calibration"]),
            "out_of_distribution_rejected": sum(not row["accepted_as_supported"] for row in rows_by_split["calibration"]),
            "out_of_distribution_rejection_rate": ood_calibration_rejection,
            "balanced_accuracy": 0.5 * (id_coverage + ood_calibration_rejection),
        },
        "audit": {},
        "sources": source_metadata,
        "sampling": {
            "seed": SEED,
            "samples_per_language_per_partition": SAMPLE_PER_PARTITION,
            "partition": "stable SHA-256 of Tatoeba sentence ID selects calibration or audit; seeded reservoir sampling within each partition",
            "overlap_check": "exact normalized texts matching in-distribution train/validation and the requested regression phrase were excluded",
        },
        "threshold": threshold,
        "final_nine_language_test_used": False,
    }
    for code in OOD_CODES:
        code_rows = [row for row in rows_by_split["audit"] if row["language_code"] == code]
        results["audit"][code] = {
            "examples": len(code_rows),
            "rejected": sum(not row["accepted_as_supported"] for row in code_rows),
            "rejection_rate": float(np.mean([not row["accepted_as_supported"] for row in code_rows])),
            "accepted_as_supported": sum(row["accepted_as_supported"] for row in code_rows),
        }

    probe_scores, probe_order, probe_top, probe_second = score_texts(
        vectorizer, classifier, [REGRESSION_PROBE]
    )
    labels = [str(label) for label in classifier.classes_]
    probe_index = probe_order[0]
    results["regression_probe"] = {
        "text": REGRESSION_PROBE,
        "true_language": "English",
        "top_prediction": labels[int(probe_index[0])],
        "top_score": float(probe_top[0]),
        "second_prediction": labels[int(probe_index[1])],
        "second_score": float(probe_second[0]),
        "all_class_scores": {
            label: float(probe_scores[0, index]) for index, label in enumerate(labels)
        },
        "rejected_by_gate": bool(probe_top[0] < threshold),
        "excluded_from_ood_calibration_and_audit": True,
    }

    (ARTIFACT_DIR / "ood_rejection_evaluation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(ARTIFACT_DIR / "ood_calibration_predictions.csv", rows_by_split["calibration"])
    write_csv(ARTIFACT_DIR / "ood_audit_predictions.csv", rows_by_split["audit"])

    print(f"OOD rejection threshold: {threshold:.9f}")
    print(f"In-distribution validation coverage: {id_coverage:.6f} ({int(np.sum(id_top >= threshold))}/{len(id_top)})")
    print(f"OOD calibration rejection: {ood_calibration_rejection:.6f} ({sum(not row['accepted_as_supported'] for row in rows_by_split['calibration'])}/{len(rows_by_split['calibration'])})")
    for code in OOD_CODES:
        metric = results["audit"][code]
        print(f"{code} audit rejection: {metric['rejection_rate']:.6f} ({metric['rejected']}/{metric['examples']})")
    print(
        f"{REGRESSION_PROBE!r}: raw top={results['regression_probe']['top_prediction']} "
        f"score={probe_top[0]:.6f}; second={results['regression_probe']['second_prediction']} "
        f"score={probe_second[0]:.6f}; rejected={results['regression_probe']['rejected_by_gate']}"
    )


if __name__ == "__main__":
    main()
