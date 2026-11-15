"""Smoke tests for the saved model and Gradio prediction function."""

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import joblib


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        example_path = (
            PROJECT_DIR
            / "artifacts"
            / "tatoeba_nine"
            / "input_output_examples.json"
        )
        cls.examples = json.loads(example_path.read_text(encoding="utf-8"))

    def test_artifacts_load_and_labels_match_examples(self):
        self.assertEqual(len(app.SUPPORTED_LANGUAGES), 9)
        self.assertTrue(app.vectorizer.vocabulary_)
        self.assertEqual(set(app.classifier.classes_), set(app.SUPPORTED_LANGUAGES))
        self.assertEqual(len(self.examples), 9)

    def test_real_example_for_each_supported_language(self):
        for example in self.examples:
            with self.subTest(language=example["true_language"]):
                prediction, score, top_three, status = app.predict_language(
                    example["text"]
                )
                self.assertIn(example["true_language"], app.SUPPORTED_LANGUAGES)
                self.assertIn(prediction, app.SUPPORTED_LANGUAGES)
                self.assertGreaterEqual(float(score), 0.0)
                self.assertLessEqual(float(score), 1.0)
                self.assertEqual(len(top_three.splitlines()), 3)
                self.assertIn("high score does not prove", status)

    def test_empty_whitespace_and_invalid_input(self):
        for value in ("", "  \n\t  "):
            with self.subTest(value=repr(value)):
                result = app.predict_language(value)
                self.assertEqual(result[:3], ("", "", ""))
                self.assertIn("Enter text", result[3])

        for value in (None, 42, object()):
            with self.subTest(value=type(value).__name__):
                result = app.predict_language(value)
                self.assertEqual(result[:3], ("", "", ""))
                self.assertIn("string", result[3])

    def test_punctuation_only_does_not_receive_a_language_prediction(self):
        prediction, score, alternatives, status = app.predict_language("!? 123 ...")
        self.assertEqual((prediction, score, alternatives), ("", "", ""))
        self.assertIn("punctuation alone", status)

    def test_short_unicode_text_is_flagged(self):
        text = self.examples[0]["text"][:5]
        prediction, score, alternatives, status = app.predict_language(text)
        self.assertIn(prediction, app.SUPPORTED_LANGUAGES)
        self.assertTrue(score)
        self.assertEqual(len(alternatives.splitlines()), 3)
        self.assertIn("very short text", status)

    def test_unsupported_language_sample_shows_closed_set_warning(self):
        # This English phrase is quoted from the user's project requirements.
        text = (
            "The application must not blindly claim that every input belongs "
            "to one of the nine languages."
        )
        prediction, score, alternatives, status = app.predict_language(text)
        self.assertEqual(prediction, app.UNASSIGNED_PREDICTION)
        self.assertIn("unsupported language", status)
        self.assertEqual(len(alternatives.splitlines()), 3)

    def test_how_are_you_is_not_assigned_to_turkish(self):
        prediction, score, top_three, status = app.predict_language("How are you?")
        self.assertEqual(prediction, app.UNASSIGNED_PREDICTION)
        self.assertLess(float(score), app.OOD_SCORE_THRESHOLD)
        self.assertIn("Turkish — model score", top_three.splitlines()[0])
        self.assertIn("raw top candidate is Turkish", status)
        self.assertIn("Rejected as uncertain", status)

    def test_real_english_and_french_audit_sentences_are_rejected(self):
        audit_path = (
            PROJECT_DIR / "artifacts" / "tatoeba_nine" / "ood_audit_predictions.csv"
        )
        with audit_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        for code in ("eng", "fra"):
            sample = next(
                row for row in rows
                if row["language_code"] == code
                and row["accepted_as_supported"] == "False"
            )
            with self.subTest(language_code=code, sentence_id=sample["sentence_id"]):
                prediction, score, _alternatives, status = app.predict_language(
                    sample["text"]
                )
                self.assertEqual(prediction, app.UNASSIGNED_PREDICTION)
                self.assertLess(float(score), app.OOD_SCORE_THRESHOLD)
                self.assertIn("Rejected as uncertain", status)

    def test_low_model_score_triggers_validation_threshold_message(self):
        low_scores = [[0.20] + [0.10] * (len(app.SUPPORTED_LANGUAGES) - 1)]
        with patch.object(app.classifier, "predict_proba", return_value=low_scores):
            prediction, score, alternatives, status = app.predict_language(
                "uncertain example"
            )
        self.assertEqual(prediction, app.UNASSIGNED_PREDICTION)
        self.assertLess(float(score), app.OOD_SCORE_THRESHOLD)
        self.assertEqual(len(alternatives.splitlines()), 3)
        self.assertIn("Rejected as uncertain", status)

    def test_long_unicode_text_is_processed(self):
        text = self.examples[0]["text"] * 40
        prediction, score, alternatives, _ = app.predict_language(text)
        self.assertIn(prediction, app.SUPPORTED_LANGUAGES)
        self.assertGreaterEqual(float(score), 0.0)
        self.assertEqual(len(alternatives.splitlines()), 3)

    def test_overlong_text_is_rejected(self):
        result = app.predict_language("a" * (app.MAX_INPUT_CHARACTERS + 1))
        self.assertEqual(result[:3], ("", "", ""))
        self.assertIn("too long", result[3])


class TatoebaArtifactTests(unittest.TestCase):
    def test_completed_pipeline_refuses_to_reuse_the_final_test_set(self):
        result = subprocess.run(
            [sys.executable, "phase_tatoeba_nine_train.py"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to rerun", result.stderr)

    def test_tatoeba_model_artifacts_load_with_required_features_and_labels(self):
        artifact_dir = PROJECT_DIR / "artifacts" / "tatoeba_nine"
        vectorizer = joblib.load(artifact_dir / "vectorizer.joblib")
        classifier = joblib.load(artifact_dir / "classifier.joblib")
        metadata = json.loads(
            (artifact_dir / "training_metadata.json").read_text(encoding="utf-8")
        )

        self.assertEqual(vectorizer.analyzer, "char")
        self.assertEqual(vectorizer.ngram_range, (1, 3))
        self.assertEqual(len(classifier.classes_), 9)
        self.assertEqual(set(classifier.classes_), set(metadata["languages"]))
        self.assertNotIn("Afaan Oromo (West Central Oromo)", classifier.classes_)
        for split in ("train_per_language", "validation_per_language"):
            counts = metadata[split]
            self.assertEqual(len(set(counts.values())), 1)
        self.assertFalse(metadata["fresh_test_set_required"])

    def test_final_test_metrics_and_split_manifest_are_complete(self):
        artifact_dir = PROJECT_DIR / "artifacts" / "tatoeba_nine"
        metadata = json.loads(
            (artifact_dir / "training_metadata.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (artifact_dir / "metrics.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (artifact_dir / "split_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(metrics["test"]["per_language"]), set(app.SUPPORTED_LANGUAGES)
        )
        self.assertEqual(len(metrics["test"]["confusion_matrix"]), 9)
        failures = json.loads(
            (artifact_dir / "failure_cases.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["test"]["failure_count"], len(failures))
        self.assertEqual(
            metrics["data"]["normalized_text_overlap_between_splits"],
            {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
        )

        split_ids = {
            split: {item["sentence_id"] for item in manifest[split]}
            for split in ("train", "validation", "test")
        }
        self.assertEqual(split_ids["train"] & split_ids["validation"], set())
        self.assertEqual(split_ids["train"] & split_ids["test"], set())
        self.assertEqual(split_ids["validation"] & split_ids["test"], set())
        self.assertEqual(len(split_ids["test"]), metrics["test"]["test_examples"])

        earlier_test_ids = set()
        previous_test_files = (
            PROJECT_DIR / "artifacts" / "tatoeba" / "test_predictions.csv",
            PROJECT_DIR
            / "artifacts"
            / "archive"
            / "tatoeba_nine_csv_parser_run"
            / "predictions.csv",
        )
        for previous_file in previous_test_files:
            with previous_file.open(encoding="utf-8", newline="") as file:
                earlier_test_ids.update(
                    row["sentence_id"] for row in csv.DictReader(file)
                )
        self.assertTrue(split_ids["test"].isdisjoint(earlier_test_ids))

        predictions_path = artifact_dir / "predictions.csv"
        with predictions_path.open(encoding="utf-8", newline="") as file:
            predictions = list(csv.DictReader(file))
        self.assertEqual(len(predictions), metrics["test"]["test_examples"])
        self.assertTrue(all(row["langdetect_prediction"] for row in predictions))
        self.assertTrue(metadata["test_examples"] > 0)
        self.assertTrue(metadata["fresh_test_set_required"] is False)
        self.assertTrue(metadata["final_test_evaluation_completed"])
        self.assertTrue(metadata["previously_scored_test_ids_excluded"])
        self.assertEqual(set(metadata["test_per_language"].values()), {39})

    def test_ood_calibration_is_separate_from_final_test_evaluation(self):
        artifact_dir = PROJECT_DIR / "artifacts" / "tatoeba_nine"
        gate = json.loads((artifact_dir / "ood_rejection.json").read_text())
        report = json.loads(
            (artifact_dir / "ood_rejection_evaluation.json").read_text()
        )
        original_metrics = json.loads((artifact_dir / "metrics.json").read_text())
        self.assertEqual(gate["ood_languages_for_calibration"], ["eng", "fra"])
        self.assertFalse(gate["uses_final_test_set"])
        self.assertFalse(report["final_nine_language_test_used"])
        self.assertTrue(report["regression_probe"]["rejected_by_gate"])
        self.assertEqual(original_metrics["test"]["test_examples"], 351)
        calibration_path = artifact_dir / "ood_calibration_predictions.csv"
        audit_path = artifact_dir / "ood_audit_predictions.csv"
        self.assertTrue(calibration_path.is_file())
        self.assertTrue(audit_path.is_file())
        with calibration_path.open(encoding="utf-8", newline="") as file:
            calibration_rows = list(csv.DictReader(file))
        with audit_path.open(encoding="utf-8", newline="") as file:
            audit_rows = list(csv.DictReader(file))
        self.assertEqual(len(calibration_rows), 1000)
        self.assertEqual(len(audit_rows), 1000)
        calibration_ids = {row["sentence_id"] for row in calibration_rows}
        audit_ids = {row["sentence_id"] for row in audit_rows}
        self.assertTrue(calibration_ids.isdisjoint(audit_ids))
        normalize = lambda text: " ".join(text.split()).casefold()
        calibration_texts = {normalize(row["text"]) for row in calibration_rows}
        audit_texts = {normalize(row["text"]) for row in audit_rows}
        self.assertTrue(calibration_texts.isdisjoint(audit_texts))
        self.assertNotIn(normalize("How are you?"), calibration_texts | audit_texts)
        self.assertEqual(
            {row["language_code"] for row in calibration_rows}, {"eng", "fra"}
        )
        self.assertEqual(
            {row["language_code"] for row in audit_rows}, {"eng", "fra"}
        )
        self.assertAlmostEqual(
            original_metrics["test"]["accuracy"], 350 / 351, places=12
        )


if __name__ == "__main__":
    unittest.main()
