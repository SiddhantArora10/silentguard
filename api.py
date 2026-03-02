"""
SilentGuard — FastAPI Backend

Receives 2-second audio chunks from the browser, classifies them with the
AST model, and sends Telegram notifications for sounds that matter.

Flow:
  Browser mic → MediaRecorder (2s chunks) → POST here → AST → classifier.py → notifier.py
"""

import io
import numpy as np
import av
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

from classifier import classify, MODES
from name_detector import is_speech, check_for_name

app = FastAPI()

# Allow requests from any origin (GitHub Pages, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load AST model once at startup ---
# MIT Audio Spectrogram Transformer — 527 AudioSet categories (same as YAMNet)
# Pure PyTorch, no TensorFlow needed
print("Loading AST model — this takes ~30 seconds on first start...")
audio_classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    top_k=1
)
print("Model ready.")


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Decode WebM/Opus audio from the browser into a float32 numpy array at 16kHz.

    Why this is needed:
    - The browser's MediaRecorder sends WebM format with Opus encoding at 48kHz
    - The AST model expects float32 audio at 16kHz
    - So we decode → normalize → resample
    """
    container = av.open(io.BytesIO(audio_bytes))

    # Read sample rate from the container header (before decoding frames)
    src_sr = container.streams.audio[0].sample_rate  # typically 48000

    # Decode all audio frames into numpy arrays
    frames = []
    for frame in container.decode(audio=0):
        frames.append(frame.to_ndarray())

    if not frames:
        raise ValueError("No audio data found in upload")

    # Flatten to mono float32
    audio = np.concatenate(frames, axis=1).flatten().astype(np.float32)

    # Normalize: WebRTC sends int16 values (-32768 to 32767), model needs -1.0 to 1.0
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    # Resample from 48kHz → 16kHz using linear interpolation
    # (same approach as the original app.py)
    if src_sr != 16000:
        n_out = int(len(audio) * 16000 / src_sr)
        audio = np.interp(
            np.linspace(0, len(audio), n_out),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    return audio


@app.get("/health")
def health():
    """Railway pings this to check if the server is running."""
    return {"status": "ok"}


@app.post("/classify")
async def classify_audio(
    audio: UploadFile = File(...),
    mode: str = Form("Sleep")
):
    """
    Main endpoint — called every 2 seconds by the browser.

    Steps:
    1. Decode the WebM audio chunk to a numpy array
    2. Run AST — get the sound label and confidence score
    3. Run classifier.py logic — check if this sound needs an alert in this mode
    4. If the model heard speech, run Whisper to check for Siddhant's name
    5. Return JSON with the result so the frontend can update the display
    """
    audio_bytes = await audio.read()

    # Decode WebM → float32 at 16kHz
    try:
        audio_array = decode_audio(audio_bytes)
    except Exception as e:
        return {"error": f"Audio decode failed: {str(e)}", "label": "—", "confidence": 0.0, "alert_sent": False}

    # Run AST classification
    result = audio_classifier({"array": audio_array, "sampling_rate": 16000})
    label = result[0]["label"]
    confidence = float(result[0]["score"])

    # Run classifier logic — modes, thresholds, Telegram alert
    alert_sent = classify(label, confidence, mode=mode)

    # If speech detected, check for name with Whisper (lazy-loaded)
    name_detected = False
    if is_speech(label):
        name_detected = check_for_name(audio_array)
        if name_detected and not alert_sent:
            alert_sent = True

    return {
        "label": label,
        "confidence": confidence,
        "alert_sent": alert_sent,
        "name_detected": name_detected
    }
