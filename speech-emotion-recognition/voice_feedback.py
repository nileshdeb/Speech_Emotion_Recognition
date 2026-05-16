"""voice_feedback.py – Speaks an AI-style message matching the detected emotion.

Uses pyttsx3 (offline TTS, no API key needed) so the app works fully locally.
Each emotion gets a distinct, empathetic message. Speaking runs in a background
thread so it never blocks the Gradio UI.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion → voice message mapping
# Add / edit messages here freely.
# ---------------------------------------------------------------------------
EMOTION_MESSAGES: dict[str, str] = {
    "fearful":   "You sound very fearful right now. It is completely okay to feel scared sometimes. Take a slow, deep breath — you are safe.",
    "angry":     "You sound quite angry. That energy is real and valid. Try to take a deep breath and give yourself a moment to cool down.",
    "happy":     "You sound absolutely happy and joyful! That is wonderful — keep spreading that positive energy around you!",
    "sad":       "You sound sad right now. It is okay to feel down sometimes. Remember, every storm passes and brighter days are ahead.",
    "neutral":   "You sound calm and neutral. A steady mind is a powerful thing. Keep that balanced energy going.",
    "calm":      "You sound very calm and relaxed. That is a beautiful state of mind. Carry that peace with you throughout your day.",
    "disgust":   "You sound disgusted about something. It is okay to feel that way. Try to focus on what you can control and let go of the rest.",
    "surprised": "You sound surprised! Life is full of unexpected moments. Embrace the wonder and stay curious.",
}

DEFAULT_MESSAGE = "Emotion detected. Please check the results on screen for more details."


def _get_engine():
    """Create a fresh pyttsx3 engine (must be created on the calling thread)."""
    import pyttsx3  # imported lazily so the module loads even if pyttsx3 is absent
    engine = pyttsx3.init()
    # Slightly slower rate for clarity
    engine.setProperty("rate", 155)
    engine.setProperty("volume", 1.0)
    return engine


def speak_emotion(emotion: str) -> None:
    """Speak the emotion feedback message in a background thread.

    Parameters
    ----------
    emotion:
        The detected emotion label (e.g. ``"fearful"``).  Case-insensitive.
    """
    message = EMOTION_MESSAGES.get(emotion.lower(), DEFAULT_MESSAGE)

    def _speak() -> None:
        try:
            engine = _get_engine()
            engine.say(message)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS playback failed: %s", exc)

    thread = threading.Thread(target=_speak, daemon=True, name="tts-voice-feedback")
    thread.start()
    logger.info("TTS started for emotion '%s': %s", emotion, message)
