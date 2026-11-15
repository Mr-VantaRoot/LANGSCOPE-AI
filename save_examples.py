"""Save one real Tatoeba training example per supported language."""

import json
from pathlib import Path

from app import SUPPORTED_LANGUAGES, predict_language


PROJECT_DIR = Path(__file__).resolve().parent
def main():
    training_path = (
        PROJECT_DIR / "artifacts" / "tatoeba_nine" / "training_examples.json"
    )
    if training_path.is_file():
        train_rows = json.loads(training_path.read_text(encoding="utf-8"))
    else:
        # Recover one real training row per label without retraining or
        # overwriting the saved model artifacts.
        from phase_tatoeba import build_splits

        train_rows, _, _, _ = build_splits()
        train_rows = [
            row for row in train_rows if row["label"] in SUPPORTED_LANGUAGES
        ]
        sample_rows = [
            next(row for row in train_rows if row["label"] == label)
            for label in SUPPORTED_LANGUAGES
        ]
        training_path.write_text(
            json.dumps(sample_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        train_rows = sample_rows
    examples = []
    for true_language in SUPPORTED_LANGUAGES:
        row = next(
            item
            for item in train_rows
            if item["label"] == true_language
        )

        predicted, confidence, top_three, status = predict_language(row["text"])
        examples.append(
            {
                "source": "Tatoeba training split",
                "sentence_id": row["id"],
                "true_language": true_language,
                "text": row["text"],
                "predicted_language": predicted,
                "confidence": confidence,
                "top_three": top_three.splitlines(),
                "status": status,
            }
        )
        print(
            f"{true_language}: predicted={predicted}; "
            f"confidence={confidence}"
        )

    output_path = (
        PROJECT_DIR / "artifacts" / "tatoeba_nine" / "input_output_examples.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(examples)} training examples to {output_path}")


if __name__ == "__main__":
    main()
