from __future__ import annotations

import html
import logging

import gradio as gr

from config import EMOTION_COLORS, EMOTION_EMOJI, MODEL_ID
from model import SpeechEmotionRecognizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_recognizer: SpeechEmotionRecognizer | None = None


def get_recognizer() -> SpeechEmotionRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = SpeechEmotionRecognizer()
    return _recognizer


def build_status_message(message: str, *, is_error: bool = False) -> str:
    background = "#fff4f4" if is_error else "#f4fff7"
    border = "#f3c2c2" if is_error else "#b8e0c5"
    color = "#b00020" if is_error else "#1f6f3d"
    return (
        f"<div style='padding: 14px; border-radius: 12px; background: {background}; "
        f"color: {color}; border: 1px solid {border};'>{html.escape(message)}</div>"
    )


def render_emotion_chart(all_scores: dict[str, float]) -> str:
    if not all_scores:
        return """
        <div style="padding: 16px; border-radius: 12px; background: #fff4f4; color: #b00020; border: 1px solid #f3c2c2;">
            No chart data available yet.
        </div>
        """

    bars = []
    for emotion, score in all_scores.items():
        width = max(score * 100.0, 1.5 if score > 0 else 0.0)
        color = EMOTION_COLORS.get(emotion.lower(), "#4f46e5")
        label = html.escape(emotion.title())
        percentage = f"{score * 100:.2f}%"
        bars.append(
            f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 14px; color: #222;">
                    <span>{label}</span>
                    <span>{percentage}</span>
                </div>
                <div style="background: #ececec; border-radius: 999px; height: 14px; overflow: hidden;">
                    <div style="width: {width:.2f}%; height: 14px; background: {color}; border-radius: 999px;"></div>
                </div>
            </div>
            """
        )

    return f"""
    <div style="padding: 18px; border-radius: 16px; background: #fafafa; border: 1px solid #e5e5e5;">
        <div style="font-size: 18px; font-weight: 700; margin-bottom: 14px; color: #111;">
            Emotion Confidence Chart
        </div>
        {''.join(bars)}
    </div>
    """


def format_result_markdown(emotion: str, confidence: float, emoji: str) -> str:
    emotion_label = html.escape(emotion.title())
    return (
        "<div style='text-align: center; padding: 12px 8px;'>"
        f"<div style='font-size: 56px; line-height: 1;'>{emoji}</div>"
        f"<div style='font-size: 30px; font-weight: 700; margin-top: 10px;'>{emotion_label}</div>"
        f"<div style='font-size: 18px; color: #444; margin-top: 8px;'>Confidence: {confidence * 100:.2f}%</div>"
        "</div>"
    )


def build_error_response(message: str) -> tuple[str, dict[str, float], str]:
    error_markdown = build_status_message(message, is_error=True)
    error_chart = f"""
    <div style="padding: 16px; border-radius: 12px; background: #fff4f4; color: #b00020; border: 1px solid #f3c2c2;">
        {html.escape(message)}
    </div>
    """
    return error_markdown, {}, error_chart


def detect_emotion(audio_path: str | None) -> tuple[str, dict[str, float], str]:
    if not audio_path:
        return build_error_response("Please record audio or upload a file before detecting emotion.")

    try:
        recognizer = get_recognizer()
        result = recognizer.predict(audio_path)
    except Exception:
        logger.exception("Unexpected error while preparing the recognizer")
        return build_error_response("Something went wrong while preparing the model. Please try again.")

    if "error" in result:
        logger.warning("Prediction returned an error for '%s': %s", audio_path, result["error"])
        return build_error_response(
            "We couldn't analyze that audio right now. Please verify the file and try again."
        )

    emotion = str(result["emotion"])
    confidence = float(result["confidence"])
    all_scores = {
        label: float(score)
        for label, score in dict(result["all_scores"]).items()
    }
    emoji = recognizer.emoji_map.get(emotion.lower(), EMOTION_EMOJI.get("neutral", "\U0001F3B5"))

    return (
        format_result_markdown(emotion, confidence, emoji),
        all_scores,
        render_emotion_chart(all_scores),
    )


def initialize_app(progress: gr.Progress = gr.Progress()) -> str:
    progress(0.0, desc="Starting application...")
    progress(0.25, desc=f"Preparing model: {MODEL_ID}")
    recognizer = get_recognizer()
    progress(0.9, desc="Finalizing startup...")

    if recognizer.initialization_error:
        logger.error("Recognizer initialization error on startup: %s", recognizer.initialization_error)
        progress(1.0, desc="Startup failed")
        return build_status_message(
            "Model failed to load. Please check the server logs and your network connection.",
            is_error=True,
        )

    progress(1.0, desc="Ready")
    return build_status_message("Model loaded successfully. You can record or upload audio now.")


with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown(
        """
        # \U0001F3A4 Speech Emotion Recognition
        Upload a voice clip or record directly from your microphone to detect the speaker's emotional tone and view confidence scores across supported emotions.
        """
    )
    status_markdown = gr.Markdown(
        value=build_status_message("Initializing model. Please wait while the app starts up.")
    )

    with gr.Row():
        audio_input = gr.Audio(
            label="Input Audio",
            type="filepath",
            sources=["microphone", "upload"],
        )

    detect_button = gr.Button("Detect Emotion", variant="primary")

    gr.Markdown("## Results")
    result_markdown = gr.Markdown(
        value="Record or upload audio, then click **Detect Emotion** to see the prediction."
    )
    scores_label = gr.Label(label="Emotion Confidence Scores", num_top_classes=8)
    chart_html = gr.HTML(value=render_emotion_chart({}))

    detect_button.click(
        fn=detect_emotion,
        inputs=audio_input,
        outputs=[result_markdown, scores_label, chart_html],
        show_progress="full",
    )
    demo.load(
        fn=initialize_app,
        inputs=None,
        outputs=status_markdown,
        show_progress="full",
    )


if __name__ == "__main__":
    demo.queue().launch(share=False, server_port=7860)
