# Credits: data, models, libraries, and AI tools

## Dataset used by the final model

- [Tatoeba sentence downloads](https://tatoeba.org/en/downloads), per-language exports `amh`, `cmn`, `jpn`, `fin`, `kat`, `tha`, `swh`, `eus`, and `tur`. The rows contain sentence IDs, three-letter language codes, and text. Tatoeba states that its sentences are published under [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/); some sentences are also CC0. The export row counts checked for this run are in [README.md](README.md), and archive SHA-256 values are in `artifacts/tatoeba_nine/metrics.json`. Sentence IDs are kept with saved predictions for traceability.
- The English (`eng`) and French (`fra`) Tatoeba exports were additionally used only for OOD rejection calibration and a disjoint OOD audit. They were not added as classifier labels and were not used to train or evaluate the final nine-class model. Their archive hashes and sample counts are recorded in `artifacts/tatoeba_nine/ood_rejection_evaluation.json`; sampled sentence IDs, text, and scores are recorded in the OOD prediction CSVs.
- The final model uses Tatoeba only. We removed all sentence IDs found in archived test predictions, exact repeated texts inside a language, and normalized text strings shared by different language labels. The final balanced sample was 258 sentences per language.

## Legacy and archived sources

- [FLORES+ (`openlanguagedata/flores_plus`)](https://huggingface.co/datasets/openlanguagedata/flores_plus), previously accessed configurations: `amh_Ethi`, `gaz_Latn`, `cmn_Hans`, `jpn_Jpan`, `fin_Latn`, `kat_Geor`, `tha_Thai`, `swh_Latn`, `eus_Latn`, and `tur_Latn`. Its card identifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and says the benchmark should not be used as training data. Earlier model artifacts used FLORES+ dev and are archived only; they are not the current model or its evaluation.
- [AfriSwitch](https://huggingface.co/datasets/intronhealth/AfriSwitch) was not downloaded or used. Access approval was pending, and the data is not part of the current nine-language project.
- Superseded or legacy model artifacts and reports remain under `artifacts/archive/` and `artifacts/tatoeba/` for traceability. The CSV-parser run under `artifacts/archive/tatoeba_nine_csv_parser_run/` is specifically marked as superseded; its test IDs were excluded from the current run and its metrics are not current results.

## Models and baselines

- Final LANGSCOPE AI model: scikit-learn [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) using character n-grams 1–3 and [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), selected with validation data only. The saved artifacts are under `artifacts/tatoeba_nine/`. Final evaluation results are in `artifacts/tatoeba_nine/metrics.json`.
- [langdetect](https://pypi.org/project/langdetect/), version 1.0.9, was run on the exact same held-out strings as the final classifier. It supports six target languages in this evaluation; Amharic, Georgian, and Basque are marked unsupported and omitted from its accuracy denominator.
- Legacy [langid.py](https://github.com/saffsd/langid.py), version 1.1.6, was used in an archived FLORES+ experiment only. It does not evaluate the final Tatoeba model.

## Software libraries

Pinned direct dependencies are listed in `requirements.txt`.

| Package | Version | Use in repository | License/source |
| --- | --- | --- | --- |
| [scikit-learn](https://scikit-learn.org/) | 1.9.1 | Character TF-IDF, Logistic Regression, metrics | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| [Gradio](https://www.gradio.app/) | 6.32.0 | Web application | [Apache 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) |
| [joblib](https://joblib.readthedocs.io/) | 1.6.0 | Save and load model artifacts | [BSD 3-Clause](https://github.com/joblib/joblib/blob/main/LICENSE.txt) |
| [langdetect](https://pypi.org/project/langdetect/) | 1.0.9 | External final-test baseline | Apache Software License, as stated on PyPI |
| [langid](https://pypi.org/project/langid/) | 1.1.6 | Archived external baseline | BSD, as stated on PyPI |
| [Hugging Face `datasets`](https://github.com/huggingface/datasets) | 5.2.0 | Earlier FLORES+ config discovery and export | [Apache 2.0](https://github.com/huggingface/datasets/blob/main/LICENSE) |
| [NumPy](https://numpy.org/) | 2.5.4 | scikit-learn numerical dependency (observed environment version) | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [SciPy](https://scipy.org/) | 1.18.1 | scikit-learn numerical dependency (observed environment version) | [BSD 3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| [pandas](https://pandas.pydata.org/) | 3.1.1 | Hugging Face data tooling dependency (observed environment version) | [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| [PyArrow](https://arrow.apache.org/) | 26.0.1 | Hugging Face data tooling dependency (observed environment version) | [Apache 2.0](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/) | 2.4.0 | Earlier Hub authentication and dataset access (observed environment version) | [Apache 2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |

## Source-code license

No separate license has been selected for the original LANGSCOPE AI source code. Dataset and dependency licenses apply to their respective works and do not grant a license to this repository's original code.

## AI tool

- [OpenAI Codex](https://openai.com/codex/) assisted with code review, implementation, and documentation. It did not generate, translate, or synthesize dataset examples. The reported predictions were produced by the saved classifier or by `langdetect`, as identified in the evaluation artifacts.
