from __future__ import annotations

import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_TOKEN"] = ""

import html
import logging
import socket

import gradio as gr

from config import EMOTION_COLORS, EMOTION_EMOJI, MODEL_ID
from model import SpeechEmotionRecognizer
from voice_feedback import speak_emotion


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


def detect_emotion(audio_path: str | None) -> tuple[str, dict[str, float], str, str]:
    if not audio_path:
        md, scores, chart = build_error_response("Please record audio or upload a file before detecting emotion.")
        return md, scores, chart, ""

    try:
        recognizer = get_recognizer()
        result = recognizer.predict(audio_path)
    except Exception:
        logger.exception("Unexpected error while preparing the recognizer")
        md, scores, chart = build_error_response("Something went wrong while preparing the model. Please try again.")
        return md, scores, chart, ""

    if "error" in result:
        logger.warning("Prediction returned an error for '%s': %s", audio_path, result["error"])
        md, scores, chart = build_error_response(
            "We couldn't analyze that audio right now. Please verify the file and try again."
        )
        return md, scores, chart, ""

    emotion = str(result["top_emotion"])
    confidence = float(result["confidence"])
    all_scores = {
        label: float(score)
        for label, score in dict(result["all_scores"]).items()
    }
    emoji = recognizer.emoji_map.get(emotion.lower(), EMOTION_EMOJI.get("neutral", "\U0001F3B5"))

    whisper_scores = result.get("whisper_scores")
    def _format_scores(scores: dict[str, float] | None) -> str:
        if scores is None:
            return "Model unavailable"
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return " | ".join(f"{emo.title()}={score * 100:.1f}%" for emo, score in items)

    debug_text = f"🎙️ Whisper: {_format_scores(whisper_scores)}"

    # ── AI Voice Feedback ──────────────────────────────────────────────────
    # Speak an empathetic message out loud matching the detected emotion.
    # Runs in a background thread so the UI is never blocked.
    speak_emotion(emotion)
    # ──────────────────────────────────────────────────────────────────────

    return (
        format_result_markdown(emotion, confidence, emoji),
        all_scores,
        render_emotion_chart(all_scores),
        debug_text,
    )


def initialize_app(progress: gr.Progress = gr.Progress()) -> str:
    progress(0.0, desc="Starting application...")
    progress(0.25, desc="Loading Whisper model...")
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


def pick_server_port() -> int:
    requested_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as requested_socket:
        requested_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if requested_socket.connect_ex(("127.0.0.1", requested_port)) != 0:
            return requested_port

    # Fall back to any free ephemeral port if the preferred port is busy.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as fallback_socket:
        fallback_socket.bind(("127.0.0.1", 0))
        return int(fallback_socket.getsockname()[1])


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

    with gr.Accordion("🎙️ Whisper Model Scores", open=False):
        debug_textbox = gr.Textbox(label="", lines=2)

    detect_button.click(
        fn=detect_emotion,
        inputs=audio_input,
        outputs=[result_markdown, scores_label, chart_html, debug_textbox],
        show_progress="full",
    )
    demo.load(
        fn=initialize_app,
        inputs=None,
        outputs=status_markdown,
        show_progress="full",
    )


if __name__ == "__main__":
    server_port = pick_server_port()
    if server_port != int(os.getenv("GRADIO_SERVER_PORT", "7860")):
        logger.warning(
            "Port %s is busy, launching Gradio on fallback port %s.",
            os.getenv("GRADIO_SERVER_PORT", "7860"),
            server_port,
        )
    demo.queue().launch(share=False, server_port=server_port)
