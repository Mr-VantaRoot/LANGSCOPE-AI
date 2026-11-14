# Credits, datasets, models, libraries, and AI tools

## Dataset

- [FLORES+ (`openlanguagedata/flores_plus`)](https://huggingface.co/datasets/openlanguagedata/flores_plus), configurations `amh_Ethi`, `gaz_Latn`, `cmn_Hans`, `jpn_Jpan`, `fin_Latn`, `kat_Geor`, `tha_Thai`, `swh_Latn`, `eus_Latn`, and `tur_Latn`. The dataset card identifies the license as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). We use its `dev` and `devtest` text and did not add generated, translated, or synthesized examples.

## Models

- LANGSCOPE AI classifier: trained in this repository using scikit-learn's character n-gram TF-IDF vectorizer and logistic regression implementation. This is a project-trained model, not a downloaded pretrained model. Its training material is the FLORES+ `dev` split described above. Model files are `artifacts/vectorizer.joblib` and `artifacts/classifier.joblib`.
- [langid.py](https://github.com/saffsd/langid.py), version 1.1.6, used only as an external evaluation baseline. Its package listing identifies a BSD license: [PyPI project page](https://pypi.org/project/langid/). Afaan Oromo is not in its supported language list, so it is reported as unsupported rather than scored as an error.

## Software libraries

| Package | Version used | Purpose | License/source |
| --- | --- | --- | --- |
| [Hugging Face `datasets`](https://github.com/huggingface/datasets) | 5.2.0 | Access and export FLORES+ | [Apache 2.0](https://github.com/huggingface/datasets/blob/main/LICENSE) |
| [scikit-learn](https://scikit-learn.org/) | 1.9.1 | TF-IDF features, logistic regression, and evaluation metrics | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| [SciPy](https://scipy.org/) | 1.18.1 | Numerical dependency used by scikit-learn | [BSD 3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| [Gradio](https://www.gradio.app/) | 6.32.0 | Web interface | [Apache 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) |
| [joblib](https://joblib.readthedocs.io/) | 1.6.0 | Save and load model artifacts | [BSD 3-Clause](https://github.com/joblib/joblib/blob/main/LICENSE.txt) |
| [langid](https://pypi.org/project/langid/) | 1.1.6 | External benchmark baseline | BSD, as stated on its PyPI page |

The pinned runtime dependencies are listed in `requirements.txt`. The environment also reported SciPy 1.18.1; scikit-learn uses SciPy as a dependency.

## AI tool

- [OpenAI Codex](https://openai.com/codex/) was used as a coding assistant to help write and document the project scripts. It did not generate or supply any dataset examples, labels, or model predictions. All reported model predictions were produced by the trained classifier or the named `langid` baseline.
