"""
name_detector.py — Detects when someone calls Siddhant's name.

How it works:
1. YAMNet tells us if it's hearing speech (not music, not noise — actual talking)
2. If yes, Whisper transcribes those 2 seconds of audio into text
3. We check if "siddhant" appears in the text
4. If yes, send a Telegram alert
"""

import whisper
import numpy as np
from notifier import send_alert

# Load Whisper's smallest model — fast enough for real-time use
# This downloads ~150MB the first time, then cached locally
whisper_model = whisper.load_model("tiny")

# The name we're listening for — exact and phonetic variants
# Whisper sometimes mishears Indian names, so we catch common transcriptions
NAME_VARIANTS = ["siddhant", "siddhan", "sidhant", "sid hunt", "siddha",'mini god','minigod','yourboisid','sid']

# YAMNet labels that mean someone is talking
SPEECH_LABELS = [
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
]


def is_speech(label):
    """
    Returns True if YAMNet thinks it's hearing speech.
    We only run Whisper when there's actual talking — saves processing time.
    """
    for speech_label in SPEECH_LABELS:
        if speech_label.lower() in label.lower():
            return True
    return False


def check_for_name(audio_data):
    """
    Takes raw audio (numpy array at 16kHz), transcribes it with Whisper,
    and checks if Siddhant's name was said.
    Returns True if the name was detected, False otherwise.
    """
    # Whisper needs float32 audio
    audio = audio_data.astype(np.float32)

    # initial_prompt primes Whisper to expect this name — improves accuracy significantly
    result = whisper_model.transcribe(audio, fp16=False, initial_prompt="My name is Siddhant.")
    text = result["text"].strip().lower()

    print(f"[whisper] heard: '{text}'")

    # Check exact name and phonetic variants (Whisper sometimes mishears Indian names)
    for variant in NAME_VARIANTS:
        if variant in text:
            print(f"[NAME DETECTED] Heard '{variant}' in: '{text}'")
            send_alert("Someone is calling your name!", confidence=1.0)
            return True

    return False
