from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE
from model import SpeechEmotionRecognizer


TEST_CASES = [
    ("Standard 440Hz tone", "test_audio.wav", 3.0, False),
    ("Very short audio", "test_audio_short.wav", 0.5, False),
    ("Very long audio", "test_audio_long.wav", 60.0, False),
    ("Silent audio", "test_audio_silent.wav", 3.0, True),
]


def generate_sine_wave(duration_seconds: float, frequency_hz: float = 440.0) -> np.ndarray:
    sample_count = int(SAMPLE_RATE * duration_seconds)
    time_axis = np.linspace(0.0, duration_seconds, sample_count, endpoint=False)
    waveform = 0.25 * np.sin(2.0 * np.pi * frequency_hz * time_axis)
    return waveform.astype(np.float32)


def generate_silent_audio(duration_seconds: float) -> np.ndarray:
    sample_count = int(SAMPLE_RATE * duration_seconds)
    return np.zeros(sample_count, dtype=np.float32)


def write_audio_file(path: Path, duration_seconds: float, silent: bool) -> None:
    audio = generate_silent_audio(duration_seconds) if silent else generate_sine_wave(duration_seconds)
    sf.write(path, audio, SAMPLE_RATE)


def format_progress_bar(score: float, width: int = 30) -> str:
    clamped_score = min(max(score, 0.0), 1.0)
    filled = int(round(clamped_score * width))
    return "#" * filled + "-" * (width - filled)


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode("unicode_escape").decode("ascii")


def validate_prediction_result(result: dict[str, Any]) -> tuple[bool, str]:
    required_keys = {"emotion", "confidence", "all_scores"}
    if not required_keys.issubset(result):
        missing = ", ".join(sorted(required_keys - set(result)))
        return False, f"missing keys: {missing}"

    confidence = result["confidence"]
    if not isinstance(confidence, (int, float)):
        return False, "confidence is not numeric"
    if not 0.0 <= float(confidence) <= 1.0:
        return False, "confidence is outside the range 0 to 1"

    all_scores = result["all_scores"]
    if not isinstance(all_scores, dict) or not all_scores:
        return False, "all_scores is empty or not a dictionary"

    for emotion, score in all_scores.items():
        if not isinstance(emotion, str):
            return False, "all_scores contains a non-string emotion label"
        if not isinstance(score, (int, float)):
            return False, f"score for {emotion!r} is not numeric"

    return True, "prediction returned the expected structure"


def print_prediction(result: dict[str, Any], recognizer: SpeechEmotionRecognizer) -> None:
    emotion = str(result["emotion"])
    confidence = float(result["confidence"])
    emoji = recognizer.emoji_map.get(emotion.lower(), "N/A")
    display_emoji = safe_console_text(emoji)

    print(f"Predicted emotion : {emotion} {display_emoji}")
    print(f"Confidence        : {confidence * 100:.2f}%")
    print("Emotion scores:")

    for label, score in result["all_scores"].items():
        numeric_score = float(score)
        bar = format_progress_bar(numeric_score)
        print(f"  {label:<10} [{bar}] {numeric_score * 100:6.2f}%")


def run_test_case(
    recognizer: SpeechEmotionRecognizer,
    name: str,
    path: Path,
    duration_seconds: float,
    silent: bool,
) -> tuple[bool, str]:
    write_audio_file(path, duration_seconds, silent)
    print(f"\n=== {name} ===")
    print(f"File              : {path.name}")
    print(f"Duration          : {duration_seconds:.2f} seconds")

    try:
        processed_audio = recognizer.preprocess_audio(str(path))
    except Exception as exc:
        return False, f"preprocess_audio failed: {exc}"

    if len(processed_audio) != recognizer.MAX_AUDIO_LENGTH:
        return (
            False,
            f"preprocess_audio returned {len(processed_audio)} samples instead of {recognizer.MAX_AUDIO_LENGTH}",
        )

    result = recognizer.predict(str(path))
    if "error" in result:
        return False, result["error"]

    is_valid, reason = validate_prediction_result(result)
    if not is_valid:
        return False, reason

    print_prediction(result, recognizer)
    return True, reason


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    recognizer = SpeechEmotionRecognizer()

    if recognizer.initialization_error:
        print("Model initialization warning:")
        print(f"  {recognizer.initialization_error}")

    results: list[tuple[str, bool, str]] = []

    for name, filename, duration_seconds, silent in TEST_CASES:
        path = output_dir / filename
        passed, reason = run_test_case(recognizer, name, path, duration_seconds, silent)
        status = "PASS" if passed else "FAIL"
        print(f"Result            : {status} - {reason}")
        results.append((name, passed, reason))

    total_passed = sum(1 for _, passed, _ in results if passed)
    print("\n=== Summary ===")
    for name, passed, reason in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:<4} {name}: {reason}")
    print(f"\nPassed {total_passed}/{len(results)} tests.")


if __name__ == "__main__":
    main()
