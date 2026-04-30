# Speech Emotion Recognition

## Overview
This project is a Gradio-based web app for detecting emotion from speech audio. You can either record directly from your microphone or upload an audio file, and the app will predict the most likely emotion, show a confidence score, and display a full confidence breakdown across the supported emotion classes.

The app uses a Hugging Face speech emotion recognition model built on top of Whisper Large V3. Audio is preprocessed to a consistent 16 kHz mono format, trimmed for silence, and normalized into a fixed-length input before inference.

## What It Does
- Accepts audio from a microphone recording or uploaded file
- Preprocesses audio for model inference
- Predicts the most likely speech emotion
- Displays the top emotion with an emoji and confidence percentage
- Shows confidence scores for all supported emotions
- Renders an inline emotion bar chart in the web interface

## Requirements
- Python 3.9 or newer
- `pip`

## Installation

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd speech-emotion-recognition
```

### 2. Create a virtual environment
On macOS or Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Setup Script
If you are on macOS or Linux, you can run:

```bash
bash setup.sh
```

This creates a virtual environment named `venv`, installs the required packages, and prints the command to start the app.

## How to Run
Start the Gradio app with:

```bash
python app.py
```

The app launches locally on port `7860`.

## How to Use

### Option 1: Microphone Recording
1. Open the app in your browser.
2. Use the audio input to record speech from your microphone.
3. Click `Detect Emotion`.
4. Wait for the prediction, confidence score, and chart to appear.

### Option 2: File Upload
1. Open the app in your browser.
2. Upload an audio file using the same audio input component.
3. Click `Detect Emotion`.
4. Review the predicted emotion and all confidence scores.

For best results:
- Use a clear voice recording
- Minimize background noise
- Keep speech reasonably short and audible

## Model Used
This project uses the Hugging Face model:

`firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`

It is loaded with:
- `AutoModelForAudioClassification`
- `AutoFeatureExtractor` with `do_normalize=True`

### Why This Model Was Chosen
This model was chosen because it combines the Whisper Large V3 backbone with fine-tuning for speech emotion recognition. According to the model card, it was trained on multiple speech emotion datasets including RAVDESS, SAVEE, TESS, and URDU. That makes it a strong choice for more varied, real-world voice samples than a smaller or narrowly trained model.

In practical terms, this helps because:
- Whisper-based models are strong at handling diverse speech audio
- The model is fine-tuned specifically for emotion classification
- It supports multiple common emotional states used in voice analysis

## Supported Emotions
The current model predicts these 7 emotions:
- Angry
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Troubleshooting

### PyAudio Installation
`pyaudio` can be the trickiest dependency, especially for microphone support.

#### Windows
Try the normal install first:
```powershell
pip install pyaudio
```

If that fails, install Microsoft C++ Build Tools or use a prebuilt wheel that matches your Python version and system architecture, then rerun:
```powershell
pip install pyaudio
```

#### macOS
Install PortAudio first:
```bash
brew install portaudio
pip install pyaudio
```

#### Linux
On Ubuntu or Debian-based systems:
```bash
sudo apt update
sudo apt install portaudio19-dev python3-dev
pip install pyaudio
```

On Fedora:
```bash
sudo dnf install portaudio-devel python3-devel
pip install pyaudio
```

If microphone support is not essential, the app can still be used with uploaded audio files as long as the rest of the dependencies install correctly.

### CUDA vs CPU Usage
The app automatically uses CUDA if PyTorch detects a compatible GPU. Otherwise it falls back to CPU.

Things to check:
- Make sure you installed a CUDA-compatible version of PyTorch if you want GPU acceleration
- Verify GPU support with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Notes:
- `True` means GPU inference is available
- `False` means the app will run on CPU
- CPU mode works, but inference will usually be slower

### Audio Sample Rate Issues
The app preprocesses audio to `16000 Hz` mono using `librosa`, so mismatched input sample rates are usually handled automatically.

If you still see issues:
- Re-export the audio as WAV if the original format is unusual
- Use a clean recording with clear speech
- Avoid extremely short or fully silent files
- Confirm the file is readable by common audio tools

The preprocessing step also trims silence and pads or truncates the clip to 30 seconds, so very long or very quiet clips may affect results.

## Notes
- The web UI runs locally with `share=False`
- Default server port is `7860`
- The model may take time to download the first time you run the app
