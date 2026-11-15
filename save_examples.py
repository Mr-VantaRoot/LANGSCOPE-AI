"""Save one real Tatoeba training example per supported language."""

import json
from pathlib import Path

from app import SUPPORTED_LANGUAGES, predict_language


PROJECT_DIR = Path(__file__).resolve().parent
def main():
    training_path = (
        PROJECT_DIR / "artifacts" / "tatoeba_ten" / "training_examples.json"
    )
    if not training_path.is_file():
        raise FileNotFoundError(
            "Saved current-model training examples are missing. Recreate the "
            "model and its split artifacts with phase_tatoeba_ten_train.py."
        )
    train_rows = json.loads(training_path.read_text(encoding="utf-8"))
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
        PROJECT_DIR / "artifacts" / "tatoeba_ten" / "input_output_examples.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(examples)} training examples to {output_path}")


if __name__ == "__main__":
    main()
