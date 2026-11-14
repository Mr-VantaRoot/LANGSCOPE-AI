"""Save real FLORES+ dev sentences with LANGSCOPE AI app outputs."""

import json
from pathlib import Path

from app import predict_language


PROJECT_DIR = Path(__file__).resolve().parent
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
    examples = []
    for true_language, config in LANGUAGES:
        path = PROJECT_DIR / "data" / "flores_plus" / "dev" / f"{config}.jsonl"
        with path.open(encoding="utf-8") as data_file:
            row = json.loads(next(data_file))

        predicted, confidence, top_three, status = predict_language(row["text"])
        examples.append(
            {
                "source": "FLORES+ dev",
                "config": config,
                "true_language": true_language,
                "text": row["text"],
                "predicted_language": predicted,
                "confidence": confidence,
                "top_three": top_three.splitlines(),
                "status": status,
            }
        )
        print(
            f"{config}: true={true_language}; predicted={predicted}; "
            f"confidence={confidence}"
        )

    output_path = PROJECT_DIR / "artifacts" / "input_output_examples.json"
    output_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(examples)} real examples to {output_path}")


if __name__ == "__main__":
    main()
