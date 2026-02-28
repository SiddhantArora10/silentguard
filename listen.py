"""
listen.py — Real-time sound classification using YAMNet
Captures mic input every 2 seconds and prints what it hears.

Run with: python listen.py
Stop with: Ctrl+C
"""

import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import csv
import io
import urllib.request

# --- Config ---
SAMPLE_RATE = 16000   # Hz — YAMNet requires this
DURATION = 2          # seconds of audio to capture per cycle
TOP_N = 3             # how many predictions to show each time

# --- Load YAMNet model ---
print("Loading YAMNet model...")
model = hub.load('https://tfhub.dev/google/yamnet/1')
print("Model loaded.")

# --- Load class names ---
print("Loading sound class names...")
class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
response = urllib.request.urlopen(class_map_url)
class_names = []
reader = csv.DictReader(io.StringIO(response.read().decode('utf-8')))
for row in reader:
    class_names.append(row['display_name'])
print(f"Ready. Listening... (Ctrl+C to stop)\n")

# --- Main loop: capture → classify → print → repeat ---
while True:
    # Capture audio from mic
    # sd.rec() records DURATION seconds at SAMPLE_RATE
    # channels=1 means mono (one mic, not stereo)
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # wait until the recording is done

    # YAMNet expects a 1D array, but sd.rec() gives us shape (samples, 1)
    # .flatten() squishes it from 2D to 1D
    audio_flat = audio.flatten()

    # Run YAMNet
    scores, embeddings, spectrogram = model(audio_flat)

    # Average scores across all frames, find the top predictions
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:TOP_N]

    # Print results
    top_label = class_names[top_indices[0]]
    top_confidence = mean_scores[top_indices[0]]
    print(f"Heard: {top_label} ({top_confidence:.0%})", end="")

    # Show runner-up if it's reasonably confident
    if len(top_indices) > 1 and mean_scores[top_indices[1]] > 0.1:
        print(f"  |  also: {class_names[top_indices[1]]} ({mean_scores[top_indices[1]]:.0%})", end="")

    print()  # newline
