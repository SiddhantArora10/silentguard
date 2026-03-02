---
title: SilentGuard
emoji: 🔔
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# SilentGuard 🔔

**AI-powered sound awareness for the hearing-impaired.**

---

## The Problem

I'm hearing-impaired. Every night when I take off my hearing aids, the world goes completely silent.

No alarms. No doorbells. No one calling my name. If something happens — a fire alarm, someone knocking — I won't hear it.

I built SilentGuard to fix that.

---

## What It Does

SilentGuard listens to your environment using your microphone and uses AI to classify sounds in real-time. When it detects something important — a doorbell, fire alarm, or knock — it instantly sends a Telegram notification to your phone.

**You don't need to hear it. Your phone will tell you.**

### Features

- **Real-time sound classification** — YAMNet classifies 521 sound categories from your mic every 2 seconds
- **Smart alerts** — only triggers above 40% confidence, and knock-type sounds need 2 consecutive detections
- **Name detection** — Whisper speech-to-text listens for your name being called, sends an alert when heard
- **3 listening modes** — switch between Sleep, Focus, and Music depending on your situation
- **Telegram notifications** — instant phone alert with the sound name and mode label

### Listening Modes

| Mode | When to use | What it alerts on |
|------|-------------|-------------------|
| **Sleep** | Hearing aids off at night | Everything — alarms, knocks, doorbells, baby cry, glass break |
| **Focus** | Working or studying | Critical only — fire alarm, smoke detector, carbon monoxide |
| **Music** | Listening to music at home | Urgent + social — alarms + doorbell + knocking |

---

## How It Works

```
Microphone → YAMNet AI → Classifier → Telegram notification
                  ↓               ↓
           Speech detected?  Streamlit dashboard
                  ↓
            Whisper STT → Name detected? → Telegram notification
```

1. Captures 2 seconds of audio from your microphone every cycle
2. Passes it through **YAMNet** — Google's audio classifier trained on 521 sound categories
3. If a match is found above 40% confidence, checks if it's a sound that matters in the current mode
4. If speech is detected, **Whisper** transcribes it and checks if your name was called
5. Sends a Telegram message to your phone instantly

---

## Tech Stack

| Tool | What it does |
|------|-------------|
| **YAMNet** | Google's pre-trained audio classifier (521 sound categories, no training needed) |
| **TensorFlow Hub** | Loads the YAMNet model |
| **Streamlit** | Live dashboard showing what's being heard |
| **Telegram Bot API** | Sends phone alerts when sounds are detected |
| **soundfile** | Reads browser mic audio for cloud deployment |
| **Whisper (OpenAI)** | Speech-to-text for name detection |
| **Python** | Everything is Python |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/SiddhantArora10/silentguard.git
cd silentguard
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Telegram bot

1. Open Telegram and message **@BotFather**
2. Create a new bot with `/newbot` — save the token it gives you
3. Message your new bot once, then visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Copy your `chat_id` from the response

### 5. Create your `.env` file

```
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## Demo

The dashboard shows what SilentGuard is hearing in real-time:

- **Currently Hearing** — the sound YAMNet detected, with confidence score
- **Recent Alerts** — log of everything it alerted you about

When a critical sound is detected, a Telegram message arrives on your phone instantly.

---

## Built By

**Siddhant Arora** — BTech CSE + Data Science, PEC Chandigarh

I'm hearing-impaired. I built this because I needed it. Most accessibility tools are built by people who imagine the problem. I live it.

- GitHub: [SiddhantArora10](https://github.com/SiddhantArora10)
- LinkedIn: [siddhant-arora-10fcb](https://www.linkedin.com/in/siddhant-arora-10fcb/)
- Email: aurorasiddhant10@gmail.com
