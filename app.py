"""
app.py — SilentGuard Streamlit UI (Cloud Version)
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import soundfile as sf
import io
import tensorflow_hub as hub
import csv
import urllib.request
from datetime import datetime
from classifier import classify, MODES
from name_detector import is_speech, check_for_name

# --- Page config ---
st.set_page_config(page_title="SilentGuard", page_icon="🔔", layout="centered")

# --- Load YAMNet model once (cached so it doesn't reload on every rerun) ---
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/yamnet/1')

@st.cache_resource
def load_class_names():
    url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    response = urllib.request.urlopen(url)
    class_names = []
    reader = csv.DictReader(io.StringIO(response.read().decode('utf-8')))
    for row in reader:
        class_names.append(row['display_name'])
    return class_names

model = load_model()
class_names = load_class_names()

# --- Session state: persists across reruns ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "current_sound" not in st.session_state:
    st.session_state.current_sound = "Waiting for audio..."
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.0

# --- UI ---
st.title("🔔 SilentGuard")
st.caption("Your AI hearing assistant — listening for sounds that matter.")
st.markdown("---")

# --- Mode selector ---
mode = st.selectbox(
    "Listening Mode",
    options=list(MODES.keys()),
    format_func=lambda m: f"{m} — {MODES[m]['description']}"
)

# --- Show current detection (rendered from session_state) ---
st.subheader("Currently Hearing")
st.metric(
    label=st.session_state.current_sound,
    value=f"{st.session_state.current_confidence:.0%}"
)

st.subheader("Recent Alerts")
if st.session_state.alerts:
    st.table(st.session_state.alerts)
else:
    st.caption("No alerts yet — all quiet.")

# --- Mic input from browser ---
# st.audio_input captures audio from the user's browser mic.
# Works on desktop and mobile — no laptop mic needed.
SAMPLE_RATE = 16000  # YAMNet requires 16kHz audio

st.markdown("---")
audio_bytes = st.audio_input("🎤 Press to record, then press stop when done")

if audio_bytes is not None:
    # Read the WAV audio that the browser recorded
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes.read()))

    # Convert stereo to mono if needed (YAMNet expects single channel)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz if needed (browser may record at 44100Hz or 48000Hz)
    if sample_rate != SAMPLE_RATE:
        num_samples = int(len(audio_data) * SAMPLE_RATE / sample_rate)
        audio_flat = np.interp(
            np.linspace(0, len(audio_data), num_samples),
            np.arange(len(audio_data)),
            audio_data
        ).astype('float32')
    else:
        audio_flat = audio_data.flatten().astype('float32')

    # --- Run YAMNet on the audio ---
    scores, embeddings, spectrogram = model(audio_flat)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_indices = np.argsort(mean_scores)[::-1]

    top_label = class_names[top_indices[0]]
    top_confidence = float(mean_scores[top_indices[0]])

    # --- Update session state with new detection ---
    st.session_state.current_sound = top_label
    st.session_state.current_confidence = top_confidence

    # --- Run classifier — sends Telegram if needed ---
    alert_sent = classify(top_label, top_confidence, mode=mode)

    # --- Name detection — runs Whisper only when YAMNet hears speech ---
    name_alert_label = None
    if is_speech(top_label):
        name_found = check_for_name(audio_flat)
        if name_found and not alert_sent:
            alert_sent = True
            name_alert_label = "Someone called your name!"

    # --- Log the alert ---
    if alert_sent:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.alerts.insert(0, {
            "Time": timestamp,
            "Sound": name_alert_label if name_alert_label else top_label,
            "Confidence": f"{top_confidence:.0%}"
        })
        st.session_state.alerts = st.session_state.alerts[:10]

    # Rerun to refresh the display with updated results
    st.rerun()
