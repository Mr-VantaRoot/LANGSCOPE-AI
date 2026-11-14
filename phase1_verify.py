"""Verify FLORES+ language data for LANGSCOPE AI. This script does not train."""

from datasets import load_dataset


DATASET_ID = "openlanguagedata/flores_plus"
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
SEED = 42


def sentence_texts(split):
    """Return non-empty text values from one dataset split."""
    return [
        str(row["text"])
        for row in split
        if row.get("text") is not None and str(row["text"]).strip()
    ]


def main():
    loaded = []

    # Load sequentially and stop on the first error, as required by the project rules.
    for label, config in LANGUAGES:
        print(f"\n=== {label} [{config}] ===", flush=True)
        try:
            dataset = load_dataset(DATASET_ID, config, token=True)
        except Exception as error:
            print(
                f"STOP: failed to load {label} [{config}]: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return 1

        loaded.append((label, config, dataset))
        print(f"Splits: {list(dataset.keys())}")

        # Print the schema and row count for every split.
        for split_name, split in dataset.items():
            print(
                f"{split_name}: columns={split.column_names}; sentences={len(split)}"
            )

        # Keep samples to three per language, using dev when available.
        sample_split = dataset["dev"] if "dev" in dataset else next(iter(dataset.values()))
        print("Three sample sentences (dev split when available):")
        for index, row in enumerate(sample_split.select(range(min(3, len(sample_split))))):
            print(f"  sample {index + 1}: {row.get('text')}")

    print("\n=== EMPTY ROWS AND DUPLICATES ===")
    for label, config, dataset in loaded:
        print(f"{label} [{config}]")
        split_names = list(dataset.keys())
        for split_name, split in dataset.items():
            texts = [row.get("text") for row in split]
            empty_count = sum(
                text is None or not str(text).strip() for text in texts
            )
            clean_texts = sentence_texts(split)
            duplicate_excess = len(clean_texts) - len(set(clean_texts))
            print(
                f"  {split_name}: empty_rows={empty_count}; "
                f"duplicate_excess_rows={duplicate_excess}"
            )

        # Count exact text matches between each pair of splits.
        for left_index, left_name in enumerate(split_names):
            for right_name in split_names[left_index + 1 :]:
                left_texts = set(sentence_texts(dataset[left_name]))
                right_texts = sentence_texts(dataset[right_name])
                matching_rows = sum(text in left_texts for text in right_texts)
                matching_unique = len(left_texts.intersection(right_texts))
                print(
                    f"  exact_duplicates_{left_name}_{right_name}: "
                    f"matching_rows={matching_rows}; "
                    f"matching_unique_sentences={matching_unique}"
                )

    # Propose train/validation from dev and reserve devtest for the final test.
    print("\n=== PROPOSED SPLIT AND BALANCE COUNTS ===")
    proposals = []
    for label, config, dataset in loaded:
        if "dev" not in dataset or "devtest" not in dataset:
            print(
                f"STOP: {label} [{config}] lacks required dev/devtest splits; "
                f"found {list(dataset.keys())}"
            )
            return 1
        dev_count = len(dataset["dev"])
        train_count = int(dev_count * 0.8 + 0.5)
        validation_count = dev_count - train_count
        test_count = len(dataset["devtest"])
        proposals.append((label, config, train_count, validation_count, test_count))
        print(
            f"{label} [{config}]: before balancing: "
            f"train={train_count}, validation={validation_count}, test={test_count}"
        )

    if proposals:
        balanced_train = min(item[2] for item in proposals)
        balanced_validation = min(item[3] for item in proposals)
        balanced_test = min(item[4] for item in proposals)
        print(
            f"Balancing by trimming each split to its smallest class count "
            f"(seed={SEED}; 80/20 split of dev; devtest reserved for test):"
        )
        for label, config, *_ in proposals:
            print(
                f"{label} [{config}]: after balancing: "
                f"train={balanced_train}, validation={balanced_validation}, "
                f"test={balanced_test}"
            )

    print("\nPhase 1 data verification complete. No model was trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
