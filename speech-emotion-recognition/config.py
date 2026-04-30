from __future__ import annotations

MODEL_ID: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
SAMPLE_RATE: int = 16000
MAX_DURATION_SECONDS: int = 30
MAX_AUDIO_LENGTH: int = SAMPLE_RATE * MAX_DURATION_SECONDS

EMOTION_COLORS: dict[str, str] = {
    "angry": "red",
    "happy": "gold",
    "sad": "steelblue",
    "neutral": "gray",
    "fearful": "purple",
    "calm": "lightgreen",
    "disgust": "brown",
    "surprised": "orange",
}

EMOTION_EMOJI: dict[str, str] = {
    "angry": "\U0001F620",
    "calm": "\U0001F60C",
    "disgust": "\U0001F922",
    "fearful": "\U0001F628",
    "happy": "\U0001F604",
    "neutral": "\U0001F610",
    "sad": "\U0001F622",
    "surprised": "\U0001F632",
}
