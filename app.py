"""Gradio app for the trained LANGSCOPE AI language identifier."""

import json
from pathlib import Path

import gradio as gr
import joblib


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
classifier = joblib.load(ARTIFACT_DIR / "classifier.joblib")
threshold_data = json.loads(
    (ARTIFACT_DIR / "uncertainty_threshold.json").read_text(encoding="utf-8")
)
CONFIDENCE_THRESHOLD = float(threshold_data["threshold"])


def predict_language(text):
    """Return model prediction, confidence, top three, and uncertainty status."""
    if text is None or not text.strip():
        return "", "", "", "Enter text to identify its language."

    # The trained TF-IDF vectorizer and logistic regression model do the prediction.
    features = vectorizer.transform([text])
    probabilities = classifier.predict_proba(features)[0]
    ranked_indices = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )[:3]

    best_index = ranked_indices[0]
    predicted_language = str(classifier.classes_[best_index])
    confidence = float(probabilities[best_index])
    top_three = "\n".join(
        f"{classifier.classes_[index]}: {float(probabilities[index])}"
        for index in ranked_indices
    )

    if confidence < CONFIDENCE_THRESHOLD:
        status = (
            "Uncertain: confidence is below the validation-selected threshold "
            f"({CONFIDENCE_THRESHOLD})."
        )
    else:
        status = (
            "Confidence meets the validation-selected threshold "
            f"({CONFIDENCE_THRESHOLD})."
        )

    return predicted_language, str(confidence), top_three, status


with gr.Blocks(title="LANGSCOPE AI", analytics_enabled=False) as demo:
    gr.Markdown(
        "# LANGSCOPE AI\n"
        "Enter text to identify its language. Predictions come from the trained "
        "character n-gram TF-IDF and logistic regression model."
    )

    text_input = gr.Textbox(
        label="Text",
        lines=5,
        placeholder="Paste text here",
    )
    identify_button = gr.Button("Identify language", variant="primary")

    predicted_output = gr.Textbox(label="Predicted language", interactive=False)
    confidence_output = gr.Textbox(label="Confidence score", interactive=False)
    top_three_output = gr.Textbox(label="Top 3 predictions", lines=3, interactive=False)
    status_output = gr.Markdown()

    outputs = [
        predicted_output,
        confidence_output,
        top_three_output,
        status_output,
    ]
    identify_button.click(
        predict_language,
        inputs=text_input,
        outputs=outputs,
        api_name="predict_language",
    )
    text_input.submit(
        predict_language,
        inputs=text_input,
        outputs=outputs,
        api_name="predict_language_on_submit",
    )


if __name__ == "__main__":
    demo.launch()
