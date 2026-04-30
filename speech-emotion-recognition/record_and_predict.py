from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import wave

import pyaudio

from config import EMOTION_EMOJI, SAMPLE_RATE
from model import SpeechEmotionRecognizer


CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
DEFAULT_SECONDS = 5

EMOTION_ASCII_ART = {
    "angry": r"""
  >:(
 /| |\
  / \
""",
    "disgust": r"""
  .-.
 ( x )
  \_/
""",
    "fearful": r"""
  .-.
 (o o)
  |=|
""",
    "happy": r"""
  \O/
   |
  / \
""",
    "neutral": r"""
  .-.
 ( - )
  ---
""",
    "sad": r"""
  .-.
 ( ; )
  /_\
""",
    "surprised": r"""
  .-.
 ( O )
  /|\
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio from your microphone and predict emotion.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=DEFAULT_SECONDS,
        help=f"Recording duration in seconds (default: {DEFAULT_SECONDS})",
    )
    return parser.parse_args()


def print_countdown() -> None:
    for count in range(3, 0, -1):
        print(f"Recording in {count}...")
        time.sleep(1)


def render_progress_bar(progress: float, width: int = 30) -> str:
    clamped = min(max(progress, 0.0), 1.0)
    filled = int(round(clamped * width))
    return "#" * filled + "-" * (width - filled)


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode("unicode_escape").decode("ascii")


def record_audio(seconds: int) -> str:
    audio = pyaudio.PyAudio()
    stream = None
    temp_path = ""

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print_countdown()
        print("Recording started...")

        frames: list[bytes] = []
        total_chunks = max(1, int(SAMPLE_RATE / CHUNK_SIZE * seconds))

        for chunk_index in range(total_chunks):
            frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))
            progress = (chunk_index + 1) / total_chunks
            bar = render_progress_bar(progress)
            print(
                f"\rRecording: [{bar}] {progress * 100:6.2f}%",
                end="",
                flush=True,
            )

        print("\nRecording finished.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(audio.get_sample_size(FORMAT))
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(b"".join(frames))

        return temp_path
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


def print_result(result: dict[str, object], recognizer: SpeechEmotionRecognizer) -> None:
    if "error" in result:
        print("\nPrediction failed:")
        print(f"  {result['error']}")
        return

    emotion = str(result["emotion"])
    confidence = float(result["confidence"])
    all_scores = {str(label): float(score) for label, score in dict(result["all_scores"]).items()}
    emoji = recognizer.emoji_map.get(emotion.lower(), EMOTION_EMOJI.get("neutral", "\U0001F3B5"))
    ascii_art = EMOTION_ASCII_ART.get(emotion.lower(), EMOTION_ASCII_ART["neutral"])

    print("\nDetected Emotion")
    print("----------------")
    print(safe_console_text(emoji), emotion.upper())
    print(ascii_art.rstrip())
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nAll scores:")

    for label, score in all_scores.items():
        bar = render_progress_bar(score)
        print(f"  {label:<10} [{bar}] {score * 100:6.2f}%")


def prompt_record_again() -> bool:
    while True:
        answer = input("\nRecord again? (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def main() -> None:
    args = parse_args()

    if args.seconds <= 0:
        print("Please provide a positive number of seconds.")
        raise SystemExit(1)

    try:
        recognizer = SpeechEmotionRecognizer()
    except Exception as exc:
        print(f"Failed to initialize recognizer: {exc}")
        raise SystemExit(1) from exc

    if recognizer.initialization_error:
        print(recognizer.initialization_error)
        raise SystemExit(1)

    while True:
        temp_path = ""
        try:
            temp_path = record_audio(args.seconds)
            print(f"Saved temporary recording to: {temp_path}")
            result = recognizer.predict(temp_path)
            print_result(result, recognizer)
        except KeyboardInterrupt:
            print("\nRecording cancelled.")
            break
        except Exception as exc:
            print(f"\nSomething went wrong: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        if not prompt_record_again():
            break


if __name__ == "__main__":
    main()
