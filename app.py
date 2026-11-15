"""Gradio interface for the trained LANGSCOPE AI language identifier."""

import json
from pathlib import Path

import gradio as gr
import joblib


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "tatoeba_nine"
vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
classifier = joblib.load(ARTIFACT_DIR / "classifier.joblib")
OOD_REJECTION_DATA = json.loads(
    (ARTIFACT_DIR / "ood_rejection.json").read_text(encoding="utf-8")
)
OOD_SCORE_THRESHOLD = float(OOD_REJECTION_DATA["threshold"])
UNASSIGNED_PREDICTION = "Uncertain — no supported language assigned"
SHORT_TEXT_CHARACTERS = 20
MAX_INPUT_CHARACTERS = 10000
SUPPORTED_LANGUAGES = tuple(str(label) for label in classifier.classes_)


def predict_language(text):
    """Return the model's top label, score, alternatives, and cautious guidance."""
    if not isinstance(text, str):
        return "", "", "", "Enter text as a string to identify its language."

    cleaned_text = text.strip()
    if not cleaned_text:
        return "", "", "", "Enter text to identify its language."

    if len(cleaned_text) > MAX_INPUT_CHARACTERS:
        return (
            "",
            "",
            "",
            f"Input is too long ({len(cleaned_text)} characters). Limit it to "
            f"{MAX_INPUT_CHARACTERS} characters.",
        )

    # Punctuation and numbers alone do not contain enough language signal to score.
    # This is input validation only; it does not choose a language.
    if not any(character.isalpha() for character in cleaned_text):
        return "", "", "", "Enter text containing letters; punctuation alone is not enough."

    try:
        features = vectorizer.transform([cleaned_text])
        scores = classifier.predict_proba(features)[0]
    except UnicodeError:
        return "", "", "", "The input contains text that could not be processed."

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:3]
    best_index = ranked_indices[0]
    predicted_language = str(classifier.classes_[best_index])
    top_score = float(scores[best_index])
    top_three = "\n".join(
        f"{classifier.classes_[index]} — model score {float(scores[index]):.6f}"
        for index in ranked_indices
    )

    messages = []
    if top_score < OOD_SCORE_THRESHOLD:
        display_language = UNASSIGNED_PREDICTION
        messages.append(
            "Rejected as uncertain: the highest supported-language score is below "
            "the OOD threshold calibrated on separate English and French validation "
            f"examples ({OOD_SCORE_THRESHOLD:.6f}). The raw top candidate is "
            f"{predicted_language}; this is not a reliable language assignment."
        )
    else:
        display_language = predicted_language
        messages.append(
            "No OOD rejection was triggered by the validation-calibrated score gate "
            f"({OOD_SCORE_THRESHOLD:.6f}). This does not prove the input is one of "
            "the supported languages."
        )

    if len(cleaned_text) < SHORT_TEXT_CHARACTERS:
        messages.append(
            f"This is very short text ({len(cleaned_text)} Unicode code points). Short "
            "text contains less character context; add more text for a more "
            "reliable result."
        )

    messages.append(
        "This model only compares the nine supported languages. It may assign an "
        "unsupported language to one of them; a high score does not prove the "
        "input belongs to a supported language."
    )
    return display_language, f"{top_score:.6f}", top_three, "\n\n".join(messages)


with gr.Blocks(
    title="LANGSCOPE AI — Language Identification",
    analytics_enabled=False,
) as demo:
    gr.Markdown(
        "# LANGSCOPE AI\n"
        "Identify text in nine languages with a character n-gram TF-IDF and "
        "logistic regression model trained on Tatoeba sentences."
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to identify",
                lines=7,
                max_lines=16,
                placeholder="Paste a sentence or more. Longer text is usually easier to identify.",
            )
            with gr.Row():
                identify_button = gr.Button("Detect Language", variant="primary")
                clear_button = gr.Button("Clear")
        with gr.Column(scale=2):
            predicted_output = gr.Textbox(
                label="Predicted language", interactive=False
            )
            score_output = gr.Textbox(
            label="Highest raw class score (not a calibrated probability)",
                interactive=False,
            )
            top_three_output = gr.Textbox(
                label="Top three candidates (model scores)",
                lines=4,
                interactive=False,
            )

    status_output = gr.Markdown()
    outputs = [predicted_output, score_output, top_three_output, status_output]
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
    clear_button.click(
        lambda: ("", "", "", "", ""),
        inputs=None,
        outputs=[text_input, *outputs],
        api_name=False,
    )

    with gr.Accordion("Supported languages and limitations", open=False):
        gr.Markdown(
            "\n".join(f"- {language}" for language in SUPPORTED_LANGUAGES)
            + "\n\nThe model always ranks these nine classes. Its score is not a "
            "guaranteed probability or a reliable test for languages outside "
            "this list. Final held-out evaluation and its limitations are described "
            "in the project README. Performance on casual "
            "messages, dialects, mixed-language text, names, and very short "
            "input has not been established."
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
