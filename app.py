"""
app.py — SilentGuard Streamlit UI
Continuous sound awareness using HuggingFace Audio Spectrogram Transformer
"""

import streamlit as st
import numpy as np
import queue
import time
from datetime import datetime

from transformers import pipeline
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode

from classifier import classify, MODES
from name_detector import is_speech, check_for_name

# --- Page config (must be first Streamlit command) ---
st.set_page_config(page_title="SilentGuard", page_icon="🔔", layout="centered")

# --- Free public STUN server for WebRTC browser connection ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Queue: background audio thread passes results to main UI thread ---
result_queue = queue.Queue(maxsize=5)

# --- Load audio classifier once (cached across reruns) ---
@st.cache_resource
def load_model():
    # MIT Audio Spectrogram Transformer — trained on 527 AudioSet categories
    # Same sound labels as YAMNet, pure PyTorch, runs natively on HF Spaces
    return pipeline(
        "audio-classification",
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        top_k=1
    )

model = load_model()


class AudioProcessor(AudioProcessorBase):
    """
    Runs in a background thread.
    Buffers browser mic audio from WebRTC, classifies every 2 seconds with AST,
    and puts the result in result_queue for the main thread to handle.
    """

    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.webrtc_sr = 48000       # WebRTC sends audio at 48kHz
        self.target_sr = 16000       # Model expects 16kHz
        self.chunk_duration = 2      # seconds per classification
        self.chunk_samples = self.webrtc_sr * self.chunk_duration

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Get raw audio from browser frame
        audio = frame.to_ndarray()

        # Convert to mono float32
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0).astype(np.float32)
        else:
            mono = audio.astype(np.float32)

        # Normalize to [-1, 1] range (WebRTC sends int16 values)
        if np.abs(mono).max() > 1.0:
            mono = mono / 32768.0

        # Accumulate frames in buffer
        self.buffer = np.concatenate([self.buffer, mono])

        # Once we have 2 seconds worth of audio, classify it
        if len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples:]

            # Resample from 48kHz → 16kHz
            n_samples = int(len(chunk) * self.target_sr / self.webrtc_sr)
            chunk_16k = np.interp(
                np.linspace(0, len(chunk), n_samples),
                np.arange(len(chunk)),
                chunk
            ).astype('float32')

            try:
                # Run audio classification
                result = model({"array": chunk_16k, "sampling_rate": self.target_sr})
                label = result[0]['label']
                confidence = float(result[0]['score'])

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

mode = st.selectbox(
    "Listening Mode",
    options=list(MODES.keys()),
    format_func=lambda m: f"{m} — {MODES[m]['description']}"
)

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

# --- WebRTC mic streamer ---
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

    # Check if background thread sent a new detection
    try:
        result = result_queue.get_nowait()
        label = result['label']
        confidence = result['confidence']
        audio = result['audio']

        st.session_state.current_sound = label
        st.session_state.current_confidence = confidence

        # Check if this sound should trigger an alert
        alert_sent = classify(label, confidence, mode=mode)

        # If speech — run Whisper to check for name
        name_alert_label = None
        if is_speech(label):
            name_found = check_for_name(audio)
            if name_found and not alert_sent:
                alert_sent = True
                name_alert_label = "Someone called your name!"

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

    # Keep the UI refreshing while listening
    time.sleep(2.5)
    st.rerun()
