# SilentGuard — Project Notes

## What Is This?

SilentGuard is an AI-powered sound awareness system I'm building because I need it.

Every night when I take off my hearing aids, I can't hear anything. Alarms, doorbells, someone knocking — all silent. SilentGuard runs in the background, listens to the environment using my laptop's microphone, and sends me a notification (Telegram or WhatsApp) when something important happens.

**This is a real tool I will actually use, not a demo project.**

---

## Core Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Sound classification | YAMNet (Google) | Pre-trained on 521 sound categories, free, open source |
| Audio capture | sounddevice (Python) | Simple real-time mic input |
| UI | Streamlit | Fast to build, easy to deploy, no frontend knowledge needed |
| Notifications | python-telegram-bot | Free, reliable, works on phone |
| Deployment | Streamlit Cloud (free) | No server cost for demos |

---

## Three Modes

| Mode | When I Use It | What Gets Notified |
|------|--------------|-------------------|
| Sleep | Bedtime (devices off) | Fire alarm, baby crying, dog bark, doorbell, knock, glass break |
| Music | When listening to music at home | Everything except ambient sounds |
| Focus | Working/studying | Important alerts only |

---

## Build Order (10-Day Plan)

- [x] Day 1: Environment setup, repo created
- [ ] Day 2: Get YAMNet running, classify one sound
- [ ] Day 3: Real-time microphone → classification → terminal output
- [ ] Day 4: Telegram notification when sound detected
- [ ] Day 5: Streamlit UI (mode selector, live alerts, history)
- [ ] Day 6: Polish + test with real sounds
- [ ] Day 7: Deploy to Streamlit Cloud, write README
- [ ] Day 8-9: LLM stretch goals (name detection, contextual alerts)

---

## Key Decisions Made

### Why YAMNet over training from scratch?
YAMNet is pre-trained on AudioSet — 521 sound categories, millions of labeled clips. Training from scratch would take weeks and significant compute. YAMNet gives us 97%+ accuracy on common sounds immediately.

### Why Streamlit over Flask/React?
Streamlit lets me build a working, visual interface with Python only. No HTML, CSS, or JavaScript needed. I can build faster and spend time on the actual problem.

### Why Telegram over WhatsApp?
Telegram has an official, free bot API. WhatsApp requires a paid business API. Telegram works on all phones and is reliable.

---

## Reference Repos (Full list in context-given-by-sarthak/reference-repos.md)

- `SangwonSUH/realtime_YAMNET` — Simple starting point for real-time YAMNet
- `robertanto/Real-Time-Sound-Event-Detection` — Most complete structure reference
- `echo-devim/pyalarmguard` — Alarm + Telegram integration example
- `makeabilitylab/ProtoSound` — CHI 2022 paper, personalizable sound recognition
- `whitphx/streamlit-webrtc` — Real-time audio in Streamlit

---

## GitHub Repo

[Add GitHub URL when created]

## Live Demo

[Add Streamlit Cloud URL when deployed]
