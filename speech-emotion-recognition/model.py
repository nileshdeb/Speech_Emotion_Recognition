from __future__ import annotations

import logging
from typing import TypedDict

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from config import CANONICAL_EMOTIONS, EMOTION_EMOJI, MAX_AUDIO_LENGTH, MAX_DURATION_SECONDS, MODEL_ID, SAMPLE_RATE


logger = logging.getLogger(__name__)


class PredictionResult(TypedDict):
    top_emotion: str
    confidence: float
    all_scores: dict[str, float]
    whisper_scores: dict[str, float] | None
    wav2vec_scores: dict[str, float] | None


class ErrorResult(TypedDict):
    error: str


PredictionResponse = PredictionResult | ErrorResult


class SpeechEmotionRecognizer:
    MODEL_ID = MODEL_ID
    SAMPLE_RATE = SAMPLE_RATE
    MAX_DURATION_SECONDS = MAX_DURATION_SECONDS
    MAX_AUDIO_LENGTH = MAX_AUDIO_LENGTH

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: AutoModelForAudioClassification | None = None
        self.feature_extractor: AutoFeatureExtractor | None = None
        self.id2label: dict[int, str] = {}
        self.emoji_map: dict[str, str] = EMOTION_EMOJI
        self.initialization_error: str | None = None

        try:
            logger.info("Loading speech emotion model '%s' on %s", self.MODEL_ID, self.device)
            self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.MODEL_ID,
                do_normalize=True,
            )
            self.id2label = {
                int(label_id): label
                for label_id, label in self.model.config.id2label.items()
            }
            logger.info("Loaded model successfully with labels: %s", list(self.id2label.values()))
        except Exception as exc:
            self.initialization_error = f"Failed to load model resources: {exc}"
            logger.exception("Model initialization failed")

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        try:
            logger.debug("Loading audio file for preprocessing: %s", audio_path)
            audio_array, _ = librosa.load(
                audio_path,
                sr=self.SAMPLE_RATE,
                mono=True,
            )
            trimmed_audio, _ = librosa.effects.trim(audio_array, top_db=20)

            if len(trimmed_audio) > self.MAX_AUDIO_LENGTH:
                processed_audio = trimmed_audio[: self.MAX_AUDIO_LENGTH]
            else:
                processed_audio = np.pad(
                    trimmed_audio,
                    (0, self.MAX_AUDIO_LENGTH - len(trimmed_audio)),
                    mode="constant",
                )

            logger.debug(
                "Preprocessed audio '%s' into %d samples at %d Hz",
                audio_path,
                len(processed_audio),
                self.SAMPLE_RATE,
            )
            return processed_audio.astype(np.float32)
        except Exception as exc:
            logger.exception("Audio preprocessing failed for '%s'", audio_path)
            raise RuntimeError(f"Failed to preprocess audio '{audio_path}': {exc}") from exc

    def _predict_whisper(self, audio_array: np.ndarray) -> dict[str, float]:
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        scores = {
            self.model.config.id2label[i]: float(p) 
            for i, p in enumerate(probabilities.tolist())
        }
        if "calm" not in scores:
            scores["calm"] = 0.0
        return scores

    def predict(self, audio_path: str) -> PredictionResponse:
        if self.initialization_error:
            logger.warning("Prediction requested while model is unavailable: %s", self.initialization_error)
            return {"error": self.initialization_error}

        if self.model is None or self.feature_extractor is None:
            logger.error("Prediction requested before model resources were ready")
            return {"error": "Model is not available for inference."}

        try:
            logger.info("Running prediction for audio file: %s", audio_path)
            audio_array = self.preprocess_audio(audio_path)

            whisper_scores = None
            try:
                whisper_scores = self._predict_whisper(audio_array)
                logger.debug("Whisper scores for '%s': %s", audio_path, whisper_scores)
            except Exception as exc:
                logger.warning("Whisper prediction failed for '%s': %s", audio_path, exc)

            if whisper_scores is None:
                raise RuntimeError("Whisper prediction failed")

            fused_scores = whisper_scores
            logger.info("Using Whisper scores for '%s'", audio_path)

            sorted_scores = sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top_emotion = sorted_scores[0][0]
            confidence = float(sorted_scores[0][1])

            result: PredictionResult = {
                "top_emotion": top_emotion,
                "confidence": confidence,
                "all_scores": dict(sorted_scores),
                "whisper_scores": whisper_scores,
                "wav2vec_scores": None,
            }
            logger.info(
                "Prediction completed for '%s': emotion=%s confidence=%.4f",
                audio_path,
                top_emotion,
                confidence,
            )
            return result
        except Exception as exc:
            logger.exception("Prediction failed for '%s'", audio_path)
            return {"error": f"Prediction failed: {exc}"}
