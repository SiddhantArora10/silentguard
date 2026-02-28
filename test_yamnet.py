"""
test_yamnet.py — Day 2 checkpoint
Load YAMNet and classify a test sound. If this works, the core of SilentGuard works.
"""

import numpy as np
import tensorflow_hub as hub
import csv
import io
import urllib.request

# --- Step 1: Load the YAMNet model from Google ---
# YAMNet is pre-trained on 521 sound categories.
# The first run downloads ~30MB from the internet.
print("Loading YAMNet model (may take a moment on first run)...")
model = hub.load('https://tfhub.dev/google/yamnet/1')
print("Model loaded!")

# --- Step 2: Load the class names (the 521 category labels) ---
# YAMNet outputs numbers (0-520). This CSV maps each number to a human name.
print("Loading sound class names...")
class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
response = urllib.request.urlopen(class_map_url)
class_map_text = response.read().decode('utf-8')

class_names = []
reader = csv.DictReader(io.StringIO(class_map_text))
for row in reader:
    class_names.append(row['display_name'])

print(f"Loaded {len(class_names)} sound categories.")

# --- Step 3: Generate a test sound ---
# We're generating a 440 Hz sine wave (the musical note A4 — like a tuning fork).
# This is just to verify YAMNet works. It should classify it as "Music" or a tone.
print("\nGenerating test audio (440 Hz sine wave, 3 seconds)...")
sample_rate = 16000          # YAMNet requires 16kHz audio
duration = 3                 # seconds
frequency = 440              # Hz — the note A4
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

# --- Step 4: Run it through YAMNet ---
print("Running YAMNet classification...")
scores, embeddings, spectrogram = model(audio)
# scores shape: (num_frames, 521) — one row per frame, one score per category

# Average scores across all frames to get overall prediction
mean_scores = np.mean(scores.numpy(), axis=0)

# --- Step 5: Print the top 5 predictions ---
top_indices = np.argsort(mean_scores)[::-1][:5]

print("\n--- Results ---")
print(f"Top predictions for the test audio:\n")
for i, idx in enumerate(top_indices):
    confidence = mean_scores[idx]
    label = class_names[idx]
    bar = "█" * int(confidence * 40)
    print(f"  {i+1}. {label:<35} {confidence:.2%}  {bar}")

print("\nYAMNet is working correctly if you see sound categories above.")
print("Checkpoint complete: model loads and classifies audio.")
