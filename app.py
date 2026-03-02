"""
app.py — SilentGuard Streamlit UI (Cloud Version with streamlit-webrtc)
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import io
import csv
import urllib.request
import queue
import time
from datetime import datetime

import tensorflow_hub as hub
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode

from classifier import classify, MODES
from name_detector import is_speech, check_for_name

# --- Page config (must be first Streamlit command) ---
st.set_page_config(page_title="SilentGuard", page_icon="🔔", layout="centered")

# --- Free public STUN server — helps browser connect to cloud server ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Queue: passes results from background audio thread to main UI thread ---
result_queue = queue.Queue(maxsize=5)

# --- Load YAMNet model once ---
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


class AudioProcessor(AudioProcessorBase):
    """
    Runs in a background thread — separate from the Streamlit UI.
    Receives audio frames from the browser mic via WebRTC.
    Buffers them until we have 2 seconds, then runs YAMNet classification.
    Puts the result in result_queue for the main thread to handle.
    """

    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.webrtc_sr = 48000      # WebRTC sends audio at 48kHz
        self.target_sr = 16000      # YAMNet needs 16kHz
        self.chunk_duration = 2     # seconds per classification cycle
        self.chunk_samples = self.webrtc_sr * self.chunk_duration

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Get raw audio from the browser frame
        audio = frame.to_ndarray()

        # Convert to mono float32
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0).astype(np.float32)
        else:
            mono = audio.astype(np.float32)

        # Normalize to [-1, 1] range (WebRTC sends int16 values)
        if np.abs(mono).max() > 1.0:
            mono = mono / 32768.0

        # Add frames to buffer
        self.buffer = np.concatenate([self.buffer, mono])

        # Once we have 2 seconds of audio, run classification
        if len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples:]

            # Resample from 48kHz to 16kHz for YAMNet
            n_samples = int(len(chunk) * self.target_sr / self.webrtc_sr)
            chunk_16k = np.interp(
                np.linspace(0, len(chunk), n_samples),
                np.arange(len(chunk)),
                chunk
            ).astype('float32')

            try:
                # Run YAMNet
                scores, _, _ = model(chunk_16k)
                mean_scores = np.mean(scores.numpy(), axis=0)
                top_idx = int(np.argsort(mean_scores)[::-1][0])

                label = class_names[top_idx]
                confidence = float(mean_scores[top_idx])

                # Send result to main thread (drop if queue is full)
                try:
                    result_queue.put_nowait({
                        'label': label,
                        'confidence': confidence,
                        'audio': chunk_16k
                    })
                except queue.Full:
                    pass

            except Exception:
                pass  # Never crash the audio thread

        return frame


# --- Session state ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "current_sound" not in st.session_state:
    st.session_state.current_sound = "Waiting to listen..."
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.0

# --- UI ---
st.title("🔔 SilentGuard")
st.caption("Your AI hearing assistant — listening for sounds that matter.")
st.markdown("---")

# Mode selector
mode = st.selectbox(
    "Listening Mode",
    options=list(MODES.keys()),
    format_func=lambda m: f"{m} — {MODES[m]['description']}"
)

# Show what's currently being heard
st.subheader("Currently Hearing")
st.metric(
    label=st.session_state.current_sound,
    value=f"{st.session_state.current_confidence:.0%}"
)

# Show alert history
st.subheader("Recent Alerts")
if st.session_state.alerts:
    st.table(st.session_state.alerts)
else:
    st.caption("No alerts yet — all quiet.")

# --- WebRTC mic streamer ---
# This widget asks the browser for mic permission and streams audio to the server.
st.markdown("---")
webrtc_ctx = webrtc_streamer(
    key="silentguard",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.state.playing:
    st.success("🟢 Listening continuously...")

    # Check if the background thread has sent a new detection result
    try:
        result = result_queue.get_nowait()
        label = result['label']
        confidence = result['confidence']
        audio = result['audio']

        # Update what's displayed
        st.session_state.current_sound = label
        st.session_state.current_confidence = confidence

        # Run classifier — sends Telegram alert if needed
        alert_sent = classify(label, confidence, mode=mode)

        # Name detection — runs Whisper only if speech was detected
        name_alert_label = None
        if is_speech(label):
            name_found = check_for_name(audio)
            if name_found and not alert_sent:
                alert_sent = True
                name_alert_label = "Someone called your name!"

        # Log the alert
        if alert_sent:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.alerts.insert(0, {
                "Time": timestamp,
                "Sound": name_alert_label if name_alert_label else label,
                "Confidence": f"{confidence:.0%}"
            })
            st.session_state.alerts = st.session_state.alerts[:10]

    except queue.Empty:
        pass

    # Rerun every 2.5 seconds to keep the display updated
    time.sleep(2.5)
    st.rerun()
