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

<!-- Add new sessions below as ## Day X sections -->
