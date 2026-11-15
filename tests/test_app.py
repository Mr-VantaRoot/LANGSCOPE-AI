"""Smoke tests for the saved model and Gradio prediction function."""

import json
import unittest
from pathlib import Path

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

    def test_unsupported_language_sample_is_marked_uncertain(self):
        # This English phrase is quoted from the user's project requirements.
        text = (
            "The application must not blindly claim that every input belongs "
            "to one of the nine languages."
        )
        prediction, score, alternatives, status = app.predict_language(text)
        self.assertIn(prediction, app.SUPPORTED_LANGUAGES)
        self.assertLess(float(score), app.CONFIDENCE_THRESHOLD)
        self.assertIn("Uncertain", status)
        self.assertIn("unsupported language", status)
        self.assertEqual(len(alternatives.splitlines()), 3)

    def test_long_unicode_text_is_processed(self):
        text = self.examples[0]["text"] * 40
        prediction, score, alternatives, _ = app.predict_language(text)
        self.assertIn(prediction, app.SUPPORTED_LANGUAGES)
        self.assertGreaterEqual(float(score), 0.0)
        self.assertEqual(len(alternatives.splitlines()), 3)


class TatoebaArtifactTests(unittest.TestCase):
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
        self.assertTrue(metadata["fresh_test_set_required"])


if __name__ == "__main__":
    unittest.main()
