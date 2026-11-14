"""Download one authorized FLORES+ split and save only sentence IDs and text."""

import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "openlanguagedata/flores_plus"
LANGUAGE_CONFIGS = [
    "amh_Ethi",
    "gaz_Latn",
    "cmn_Hans",
    "jpn_Jpan",
    "fin_Latn",
    "kat_Geor",
    "tha_Thai",
    "swh_Latn",
    "eus_Latn",
    "tur_Latn",
]
EXPECTED_SENTENCES = {"dev": 997, "devtest": 1012}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=EXPECTED_SENTENCES)
    split_name = parser.parse_args().split
    output_dir = Path("data") / "flores_plus" / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in LANGUAGE_CONFIGS:
        print(f"Loading {config} {split_name} split...", flush=True)
        dataset = load_dataset(
            DATASET_ID,
            config,
            split=split_name,
            token=True,
        )
        if len(dataset) != EXPECTED_SENTENCES[split_name]:
            raise ValueError(
                f"Expected {EXPECTED_SENTENCES[split_name]} sentences for "
                f"{config}; found {len(dataset)}"
            )

        output_path = output_dir / f"{config}.jsonl"
        dataset.select_columns(["id", "text"]).to_json(
            str(output_path),
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print(f"Saved {len(dataset)} sentences to {output_path}", flush=True)


if __name__ == "__main__":
    main()
