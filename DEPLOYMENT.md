# Deploying LANGSCOPE AI to Hugging Face Spaces

**Deployment has not been performed or tested.** These are preparation instructions only; they do not create a Space, account, or public endpoint.

Hugging Face Spaces uses a YAML metadata block at the top of the Space repository's README to select the SDK, Python version, Gradio version, and app file. See the official [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference), [Gradio Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-gradio), and [dependency guide](https://huggingface.co/docs/hub/spaces-dependencies).

## Before creating a Space

1. Run the tests in a normal local environment:

   ```bash
   python -m unittest discover -s tests -v
   ```

2. Launch `python app.py`, open the local Gradio URL, and try one saved sample per language, plus blank, punctuation-only, short, overlong, and unsupported-language text.
3. Review [README.md](README.md) and [CREDITS.md](CREDITS.md). In particular, explain that the model is closed-set and the score is not a calibrated probability.
4. Decide whether the Space should be public or private. A public Space exposes its source files and bundled model artifacts. Keep raw dataset archives out of the Space.

## Create the Space

1. Create a new Space manually on Hugging Face and select the **Gradio** SDK. Choose its visibility and hardware deliberately.
2. In the Space repository's root `README.md`, add this YAML block at the top (do not add a software license unless the team chooses one):

   ```yaml
   ---
   title: LANGSCOPE AI
   short_description: Nine-language text identification with character n-gram TF-IDF.
   sdk: gradio
   sdk_version: 6.32.0
   python_version: "3.14"
   app_file: app.py
   ---
   ```

   The platform's accepted metadata can change; check the linked configuration reference before creation.

3. Copy the following files and directories into the Space repository:

   ```text
   app.py
   requirements.txt
   CREDITS.md
   README.md
   artifacts/tatoeba_nine/classifier.joblib
   artifacts/tatoeba_nine/vectorizer.joblib
   artifacts/tatoeba_nine/ood_rejection.json
   ```

   The app loads these exact paths relative to `app.py`. Do not upload the legacy artifacts or raw `data/` exports. The metrics, OOD evaluation, split manifest, and failure cases may be included for transparency, but are not required at runtime.

4. Commit the files to the Space repository. The Space will install `requirements.txt` and start `app.py`.
5. Review build logs. After a successful build, open the Space and manually check prediction output, the short-text warning, and the low-score uncertainty message. Confirm that the share visibility is intentional before presenting the URL.

No deployment, external account creation, or changes to GitHub branches/settings have been performed.
