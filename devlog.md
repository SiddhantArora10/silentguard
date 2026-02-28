# SilentGuard — Dev Log

> A running journal of how this project was built.
> Every session: what we did, exact commands we ran, why, and what we learned.

---

## Day 1 — Feb 28, 2026 | Setup

### Goal
Get the project structure ready before writing any code.

### What we decided
- **Stack:** Python + YAMNet (Google's audio classifier) + Streamlit + Telegram notifications
- **Why YAMNet:** Pre-trained on 521 sound categories. No need to train from scratch. Free and open source.
- **Why Streamlit:** Build a visual UI using only Python. No HTML/CSS needed.
- **Why Telegram:** Has a free official bot API. WhatsApp requires a paid business API.
- **Repo structure:** `silentguard/` is a public GitHub repo nested inside the private `siddhant-space/` repo. The outer repo ignores this folder via `.gitignore`.

### Commands run
*(will be filled in as we run them)*

### Files created
*(will be filled in as we create them)*

### What's next
- [ ] Initialize git in this folder
- [ ] Install Python 3.12 (TensorFlow doesn't support 3.14 yet)
- [ ] Create virtual environment
- [ ] Install dependencies: tensorflow, sounddevice, streamlit, python-telegram-bot
- [ ] Verify YAMNet loads correctly
- [ ] Push to public GitHub repo

---

## Day 2 — Feb 28, 2026 | YAMNet + Real-time Listening

### Goal
Get YAMNet classifying audio. First from a synthetic test, then from a real microphone.

### What we built
- **`test_yamnet.py`** — loads YAMNet from TensorFlow Hub, generates a 440 Hz sine wave, classifies it, prints top 5 predictions. Passed successfully.
- **`listen.py`** — real-time loop: captures 2 seconds of mic audio, flattens it, runs through YAMNet, prints the top classification every cycle.

### Commands run
```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install tensorflow tensorflow-hub sounddevice numpy
python test_yamnet.py   # passed
python listen.py        # working from mic
```

### What's next
- [ ] Add Telegram bot — send alert when specific sound detected
- [ ] Build Streamlit UI to replace terminal output
- [ ] Define alert sounds: doorbell, alarm, knocking, dog bark

---

<!-- Add new sessions below as ## Day X sections -->
