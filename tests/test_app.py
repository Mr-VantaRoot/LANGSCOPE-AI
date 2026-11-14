"""Smoke tests for the saved model and Gradio prediction function."""

import json
import unittest
from pathlib import Path

import app


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        example_path = PROJECT_DIR / "artifacts" / "input_output_examples.json"
        cls.examples = json.loads(example_path.read_text(encoding="utf-8"))

    def test_artifacts_load_and_labels_match_examples(self):
        self.assertEqual(len(app.SUPPORTED_LANGUAGES), 10)
        self.assertTrue(app.vectorizer.vocabulary_)
        self.assertEqual(set(app.classifier.classes_), set(app.SUPPORTED_LANGUAGES))
        self.assertEqual(len(self.examples), 10)

    def test_real_example_for_each_supported_language(self):
        for example in self.examples:
            with self.subTest(language=example["true_language"]):
                prediction, score, top_three, status = app.predict_language(
                    example["text"]
                )
                self.assertEqual(prediction, example["true_language"])
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
            "to one of the ten languages."
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


if __name__ == "__main__":
    unittest.main()
