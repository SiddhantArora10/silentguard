"""
app.py — SilentGuard Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import csv
import io
import urllib.request
from datetime import datetime
from classifier import classify

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
    st.session_state.current_sound = "Starting up..."
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.0

# --- UI: render from session_state FIRST, before capturing audio ---
# This way the content is visible during the 2-second audio capture window
st.title("🔔 SilentGuard")
st.caption("Your AI hearing assistant — listening for sounds that matter.")
st.markdown("---")

st.success("🟢 Listening...")

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

# --- NOW capture audio (UI stays visible during this 2-second wait) ---
SAMPLE_RATE = 16000
DURATION = 2

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

audio_flat = audio.flatten()
scores, embeddings, spectrogram = model(audio_flat)
mean_scores = np.mean(scores.numpy(), axis=0)
top_indices = np.argsort(mean_scores)[::-1]

top_label = class_names[top_indices[0]]
top_confidence = float(mean_scores[top_indices[0]])

# --- Update session_state with new detection ---
st.session_state.current_sound = top_label
st.session_state.current_confidence = top_confidence

# --- Run classifier — sends Telegram if needed ---
alert_sent = classify(top_label, top_confidence)

# --- Log the alert ---
if alert_sent:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.alerts.insert(0, {
        "Time": timestamp,
        "Sound": top_label,
        "Confidence": f"{top_confidence:.0%}"
    })
    st.session_state.alerts = st.session_state.alerts[:10]

# --- Rerun to capture the next audio chunk ---
st.rerun()
