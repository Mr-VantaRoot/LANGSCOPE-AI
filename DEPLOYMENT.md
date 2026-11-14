# Deploying LANGSCOPE AI later

No hosted deployment has been created or tested. These steps describe a manual Hugging Face Spaces deployment; they do not create an account, Space, or public endpoint. Check the official [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) and [Gradio Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-gradio) for current account, hardware, and SDK requirements before creating one.

## Before deployment

1. Run the local checks from the repository root:

   ```bash
   python -m unittest discover -s tests -v
   python app.py
   ```

2. Confirm the browser opens and test representative inputs from all ten supported labels. Review the closed-set limitation and score wording before showing the demo.
3. Decide whether the Space should be public or private. A public Space exposes the app and its bundled model artifacts. Do not upload raw FLORES+ exports; they are unnecessary at runtime and are ignored by Git.

## Create the Space manually

1. Sign in to Hugging Face and create a new Space using the **Gradio** SDK. Choose visibility and hardware appropriate for the event.
2. Upload or push the following runtime files to the Space repository:

   ```text
   app.py
   requirements.txt
   artifacts/classifier.joblib
   artifacts/vectorizer.joblib
   artifacts/uncertainty_threshold.json
   ```

   The evaluation JSON and CSV files are not needed to launch the UI. Include `README.md` and `CREDITS.md` for project context and attribution.
3. Let the Space build from `requirements.txt`. The Gradio SDK starts the app from `app.py`; if the Space build interface asks for an app entry file, select `app.py`.
4. Check the build logs for dependency or artifact-loading errors. Open the Space URL and verify one real example for each language, blank input, punctuation-only input, and an unsupported-language example. Confirm the uncertainty and short-text notices appear as expected.
5. Share the hosted URL only after those checks pass and the visibility setting is intentional.

Deployment may require adjustments if the hosting platform changes its current SDK or build conventions. No GitHub branches, repository settings, or external accounts are changed by this project.
