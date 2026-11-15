# Credits, datasets, models, libraries, and AI tools

## Data used by the current nine-language model

- [Tatoeba sentences exports](https://tatoeba.org/en/downloads), per-language files `amh`, `cmn`, `jpn`, `fin`, `kat`, `tha`, `swh`, `eus`, and `tur`. Tatoeba documents sentence rows as sentence ID, language code, and sentence text. Its downloads page states the export is licensed under [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/); some sentences are also CC0 1.0. The verified local exports contained 314 Amharic, 89,315 Mandarin, 248,989 Japanese, 155,021 Finnish, 7,710 Georgian, 6,853 Thai, 4,583 Swahili, 6,438 Basque, and 750,286 Turkish rows. Sentence IDs are retained in saved examples for attribution and traceability.
- The Tatoeba export for West-Central Oromo (`gaz`) was checked and had 10 rows. Oromo was excluded from the model and is not a supported app language.

## Legacy data retained for traceability

- [FLORES+ (`openlanguagedata/flores_plus`)](https://huggingface.co/datasets/openlanguagedata/flores_plus), configurations previously examined: `amh_Ethi`, `gaz_Latn`, `cmn_Hans`, `jpn_Jpan`, `fin_Latn`, `kat_Geor`, `tha_Thai`, `swh_Latn`, `eus_Latn`, and `tur_Latn`. The dataset card identifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and describes FLORES+ as an evaluation benchmark that should not be used as training data. It was used in legacy artifacts and an archived experiment, but not to train the current nine-language model. Do not present those legacy results as the current model's evaluation.
- [AfriSwitch](https://huggingface.co/datasets/intronhealth/AfriSwitch) was not downloaded or used. Access is pending author approval. No AfriSwitch data, examples, or results are included in this project.

## Models and libraries

- Current app classifier: scikit-learn `TfidfVectorizer(analyzer="char", ngram_range=(1, 3))` and `LogisticRegression`, trained on Tatoeba sentences for the nine languages listed above. Saved files are in `artifacts/tatoeba_nine/`. Hyperparameters and the uncertainty threshold were selected using validation data only. No final test evaluation has been run for this model.
- Legacy FLORES+ model: preserved at `artifacts/vectorizer.joblib` and `artifacts/classifier.joblib`; trained with FLORES+ dev data. It is retained for provenance only.
- Archived ten-language Tatoeba model and results: preserved under `artifacts/tatoeba/`; that experiment included an Oromo FLORES+ supplement and already used its test set. It is not a final compliant result and must not be rerun as a final evaluation.
- [Gradio](https://www.gradio.app/) provides the web interface.
- [Hugging Face `datasets`](https://github.com/huggingface/datasets) accessed and exported FLORES+ during the earlier project work.
- [langid.py](https://github.com/saffsd/langid.py), version 1.1.6, and [langdetect](https://pypi.org/project/langdetect/), version 1.0.9, were used by archived baseline experiments. Those archived outputs do not evaluate the current nine-language model.

## Software library versions

Versions pinned in `requirements.txt`:

| Package | Version | Purpose | License/source |
| --- | --- | --- | --- |
| [datasets](https://github.com/huggingface/datasets) | 5.2.0 | Earlier FLORES+ access/export | [Apache 2.0](https://github.com/huggingface/datasets/blob/main/LICENSE) |
| [scikit-learn](https://scikit-learn.org/) | 1.9.1 | TF-IDF, Logistic Regression, and metrics | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| [SciPy](https://scipy.org/) | 1.18.1 | Numerical dependency | [BSD 3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| [NumPy](https://numpy.org/) | 2.5.4 | Numerical operations | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [pandas](https://pandas.pydata.org/) | 3.1.1 | Dataset tooling dependency | [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| [PyArrow](https://arrow.apache.org/) | 26.0.1 | Hugging Face dataset data support | [Apache 2.0](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/) | 2.4.0 | Earlier Hub access/authentication | [Apache 2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| [Gradio](https://www.gradio.app/) | 6.32.0 | Web interface | [Apache 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) |
| [joblib](https://joblib.readthedocs.io/) | 1.6.0 | Model serialization | [BSD 3-Clause](https://github.com/joblib/joblib/blob/main/LICENSE.txt) |
| [langid](https://pypi.org/project/langid/) | 1.1.6 | Archived external baseline | BSD, as stated on PyPI |
| [langdetect](https://pypi.org/project/langdetect/) | 1.0.9 | Archived external baseline | Apache Software License, as stated on PyPI |

## Source-code license

No separate license has been selected for the original project source code. Dataset terms apply to the respective datasets and are not a license grant for this repository's source code.

## AI tool

- [OpenAI Codex](https://openai.com/codex/) assisted with coding and documentation. It did not generate, translate, or synthesize dataset examples. Reported model predictions come from the trained classifier; no results for the current model's test set are claimed.
