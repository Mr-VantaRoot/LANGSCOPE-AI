# LANGSCOPE AI

**Multilingual language identification with machine learning** — built for the AI + Languages track of the VERITAS AI International High School Hackathon 2026.

## Problem and objective

Short or unfamiliar text can be hard to identify, especially for languages with limited data. LANGSCOPE AI predicts among ten languages, displays its top candidates, and can abstain when the leading score is low.

**Supported languages:** Amharic, Basque, English, Finnish, Georgian, Japanese, Mandarin Chinese, Swahili, Thai, and Turkish.

## Method

We used Tatoeba's public sentence exports (IDs, language codes, and text). After excluding previously scored IDs and duplicate or cross-language-ambiguous text, we balanced to **219 unique sentences per language** (**2,190 total**); Amharic limited the sample. Each language has 153 training, 33 validation, and 33 held-out test sentences, with no normalized exact-text overlap between splits.

The model uses `TfidfVectorizer(analyzer="char", ngram_range=(1, 3))` and Logistic Regression. Validation selected `C=10.0`; the uncertainty threshold used in-language validation and separate French out-of-distribution examples. See [CREDITS.md](CREDITS.md) for dataset and tool attribution.

## Held-out results

On the fresh test set (**330 sentences; 33 per language**), the model got **329 correct (99.70% accuracy)**, with **99.71% macro precision**, **99.70% macro recall**, and **99.70% macro F1**. The majority-class baseline was **10.00%**.

Each language has only **33 test examples**. A 100.00% accuracy or recall means all 33 were classified correctly in this sample; 100.00% precision means every test sentence predicted as that language was correct. These small-sample results do not mean the model is perfect on that language generally.

| Language | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Amharic | 100.00% | 100.00% | 100.00% | 100.00% |
| Basque | 96.97% | 100.00% | 96.97% | 98.46% |
| English | 100.00% | 100.00% | 100.00% | 100.00% |
| Finnish | 100.00% | 97.06% | 100.00% | 98.51% |
| Georgian | 100.00% | 100.00% | 100.00% | 100.00% |
| Japanese | 100.00% | 100.00% | 100.00% | 100.00% |
| Mandarin Chinese | 100.00% | 100.00% | 100.00% | 100.00% |
| Swahili | 100.00% | 100.00% | 100.00% | 100.00% |
| Thai | 100.00% | 100.00% | 100.00% | 100.00% |
| Turkish | 100.00% | 100.00% | 100.00% | 100.00% |

The only test error was Basque “Ikasi,” predicted as Finnish; the app abstains on its low score. See the [confusion matrix](artifacts/tatoeba_ten/confusion_matrix.csv), [failure case](artifacts/tatoeba_ten/failure_cases.json), and [full metrics](artifacts/tatoeba_ten/metrics.json).

**Baseline:** `langdetect` 1.0.9 scored **92.21%** on the same 231 test examples where it supports the label (70% test coverage). Amharic, Georgian, and Basque are unsupported and excluded. Per-language results are in [metrics](artifacts/tatoeba_ten/metrics.json).

## What we added

- **English support:** English is the tenth class; “How are you?” ranks English first (raw score 0.841009).
- **Short-text evaluation:** On the same test set, accuracy was 44/45 for inputs up to 10 characters, 84/84 for 11–20, and 201/201 above 20. These are small groups, not separate tests.
- **Uncertainty handling:** The app shows the three highest raw Logistic Regression scores (not calibrated probabilities). Below **0.660368**, it displays “Uncertain — no supported language assigned.” The gate retained 95.15% of in-language validation examples but rejected only 43.4% of a separate French audit, so it does not reliably detect every unsupported language. See the [OOD evaluation](artifacts/tatoeba_ten/ood_rejection_evaluation.json) and [audit predictions](artifacts/tatoeba_ten/ood_audit_predictions.csv).
- **Gradio app:** Displays the prediction, score, alternatives, and short-text or uncertainty guidance. Run with `python app.py` after installing [requirements.txt](requirements.txt); see [deployment instructions](DEPLOYMENT.md).

## Limitations

The dataset is small (219 sentences per language; 33 test examples per class) and consists of curated Tatoeba text. Results may differ on casual conversation, dialects, spelling variation, names, mixed-language text, or other domains. OOD rejection is imperfect: unsupported text may pass and supported text may be rejected. These are results from this held-out run, not a broad external benchmark.

For reproducibility and detail, see the [split manifest](artifacts/tatoeba_ten/split_manifest.json), [training metadata](artifacts/tatoeba_ten/training_metadata.json), and [all test predictions](artifacts/tatoeba_ten/predictions.csv). Tatoeba's dataset terms do not license this repository's source code; no software license has been selected.
