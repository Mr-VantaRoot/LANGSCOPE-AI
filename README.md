# LANGSCOPE AI

## Problem

Language identification can be difficult for short text and for languages that are less commonly supported by general purpose tools. LANGSCOPE AI is a small, reproducible classifier for ten languages, built for the AI + Languages track of the VERITAS AI International High School Hackathon 2026.

The current data represents West Central Oromo (`gaz_Latn`), Simplified Mandarin (`cmn_Hans`), and the named FLORES+ varieties/configurations. Results therefore describe this benchmark and these varieties; they do not establish performance across every dialect, writing style, or source.

## What we built

A Gradio app that accepts text and shows the predicted language, the model's maximum class probability, the top three classes and probabilities, and a validation-calibrated uncertainty message. Predictions come from a trained character n-gram TF-IDF vectorizer and multinomial logistic regression classifier. The app has no script-detection or keyword rules.

## What we added

- A Phase 1 data verification script and an exporter for the official FLORES+ `dev` and `devtest` splits.
- A reproducible training and validation-tuning script, with fixed seed 42.
- A one-time held-out test evaluation, comparison baselines, confusion matrix, short-prefix measurements, and real failure examples.
- A validation-only confidence threshold selection script.
- Ten real `dev` input/output examples in `artifacts/input_output_examples.json`.
- Saved model and vectorizer in `artifacts/`.

## How it works

FLORES+ supplies parallel sentences in ten configurations. The `dev` split has 997 sentences per language. A seeded split of shared sentence IDs gives 798 training and 199 validation sentences per language; `devtest` is reserved for the final test, with 1,012 per language. Phase 1 inspection found zero empty rows, zero within-split exact duplicate sentences, and zero exact sentence matches between `dev` and `devtest` for these configurations.

The training script selects settings using validation macro-F1, with validation accuracy as a tie-breaker. It compares character n-gram ranges `(1, 3)`, `(2, 4)`, `(3, 5)` and logistic regression `C` values `0.1`, `1`, and `10`. The selected settings were `(1, 3)` and `C=0.1`; all nine tested settings tied at validation accuracy and macro-F1 of 1.0. The saved final model was fit on the training split only.

The confidence threshold is chosen on validation data to maximize accuracy among accepted predictions while maintaining at least 95% coverage, with ties resolved in favor of the higher threshold. The resulting threshold is `0.4757289252041075`; validation coverage was 0.950251256281407 and selective accuracy was 1.0. Predictions below the threshold are marked uncertain. This is a closed-set classifier: it always ranks the ten known classes, and the threshold is not a general detector for languages outside those classes.

## Results

The test set was evaluated once after settings were fixed. The classifier got 10,119 of 10,120 examples correct (accuracy `0.9999011857707509`). It made one error: a Basque sentence was predicted as Simplified Mandarin. Per-language precision, recall, F1, and the full confusion matrix are in `artifacts/phase3_metrics.json` and `artifacts/confusion_matrix.csv`.

The majority-class baseline accuracy was `0.1`. `langid` 1.1.6 scored `0.9997804128238911` accuracy on its 9,108 supported test examples; Afaan Oromo was unsupported and excluded from that comparison.

Accuracy on prefixes taken from each real test sentence was:

| Prefix length (Unicode code points) | Accuracy | Examples |
| --- | ---: | ---: |
| 5 | 0.3268774703557312 | 10,120 |
| 10 | 0.6575098814229249 | 10,120 |
| 20 | 0.9354743083003952 | 10,120 |

These measurements show that very short fragments are much harder. The five saved real failure cases are in `artifacts/failure_cases.json`, with possible explanations; those explanations are interpretations, not additional measured findings. The saved app examples are illustrative `dev` records, not test results.

## What we'd do next

- Evaluate on naturally occurring user text and additional varieties, with the same strict separation of training, validation, and test data.
- Explore more robust calibration and an explicit out-of-scope response using validation-only decisions.
- Have Oromo speakers review the chosen West Central Oromo data and test examples.
- Run the Gradio app locally in a normal terminal and verify the browser interaction; the development sandbox used for this session denied socket creation, so an HTTP/browser-level check could not be completed here.

## Sources

- Dataset: [FLORES+ dataset card](https://huggingface.co/datasets/openlanguagedata/flores_plus); data license: CC BY-SA 4.0, as stated in its dataset card.
- External detector baseline: [langid.py](https://github.com/saffsd/langid.py), version 1.1.6. Its supported language list does not include Oromo (`om`).
- Library, model and tool references and licenses: see [CREDITS.md](CREDITS.md).

## Run it

Install the pinned packages and launch the UI:

```bash
python -m pip install -r requirements.txt
python app.py
```

To reproduce training and evaluation, first export the two dataset splits using the authenticated Hugging Face account with FLORES+ access, then run `phase2_train.py`, `phase4_calibrate.py`, and `phase3_evaluate.py` in that order. The test evaluation script is intended to be run once after model and threshold choices are finalized. Raw exports under `data/flores_plus/` are not committed.
