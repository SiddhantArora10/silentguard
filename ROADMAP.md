# SilentGuard — Product Roadmap

> End vision: A phone app (PWA) that any hearing-impaired person installs, sets on their bedside, and trusts to wake them up when it matters.
>
> Last updated: 2026-03-25

---

## Current State

- Browser mic → 2-second chunks → AST classifies 527 sounds → Telegram alert
- 3 modes (Sleep/Focus/Music), name detection via Whisper
- Frontend: GitHub Pages | Backend: HuggingFace Spaces (Docker)
- **Known issues:** Telegram blocked on HF, XSS in frontend, missing aria labels, Whisper lazy-load hang

---

## Phase 1: Fix What's Broken (make it reliable)

**Goal:** The demo becomes a tool you actually use every night.

- [ ] Fix or replace Telegram alerts (blocked on HF Spaces)
- [ ] Fix XSS vulnerability in `addAlertRow()` — sanitize `result.label` before innerHTML
- [ ] Add aria labels to all interactive elements (your own app should be accessible)
- [ ] Pre-load Whisper at startup (eliminate 3-minute hang on first speech detection)
- [ ] Update README completely (YAMNet → AST, Streamlit → FastAPI, Railway → HF Spaces)
- [ ] Fix dropdown arrow styling

**Interview line:** "I treat my personal project with production standards — security, accessibility, reliability."

---

## Phase 2: PWA — Make It a Phone App

**Goal:** SilentGuard on your phone, on your bedside table. No laptop needed.

- [ ] Add `manifest.json` — app name, icons, theme color, display: standalone
- [ ] Add service worker — caching, offline shell, background operation
- [ ] Web Push notifications — replaces Telegram entirely, works in background
- [ ] Vibration API — phone vibrates for critical sounds (what actually wakes you up)
- [ ] Wake Lock API — keeps mic alive overnight, prevents phone from sleeping
- [ ] Mobile-optimized UI — bigger buttons, touch-friendly, landscape bedside mode
- [ ] "Add to Home Screen" prompt — guide users to install

**Why PWA over native app?** No App Store, no Swift/Kotlin, you already know HTML/JS. PWA with push + vibration is 90% of native.

**Interview line:** "I turned a web demo into an installable phone app with push notifications and vibration — 330M people could use this."

---

## Phase 3: Near-Real-Time Detection

**Goal:** From "2 seconds" to "feels instant."

- [ ] Sliding window — classify overlapping 2-second windows every 0.5 seconds
- [ ] WebSocket connection — replace HTTP POST per chunk with persistent WebSocket
- [ ] On-device pre-screening — lightweight model in browser (TensorFlow.js) filters silence/noise, only sends interesting audio to AST backend
- [ ] Result: detection latency drops from ~2s to ~0.5s

**Interview line:** "I optimized detection latency from 2 seconds to sub-500ms using sliding windows and a two-tier model architecture."

---

## Phase 4: Multi-User Product

**Goal:** Other people can use it, not just Siddhant.

- [ ] Onboarding wizard — "What sounds matter to you?"
- [ ] Custom sound profiles — user-defined modes (Baby Room, Office, Kitchen)
- [ ] Custom alert names — any name, not just "Siddhant"
- [ ] Sound sensitivity calibration — "Clap twice to calibrate your room"
- [ ] History dashboard — sounds detected last night, weekly patterns
- [ ] User accounts (optional) — save preferences across devices

**Interview line:** "I built onboarding, custom profiles, and a sound history dashboard — product thinking, not just code."

---

## Phase 5: Dream Features (long-term)

- [ ] Custom sound training — record your doorbell 5 times, model learns it
- [ ] Smart home integration — flash Philips Hue, trigger smart plugs via IFTTT
- [ ] Wearable vibration — smartwatch wrist-tap alerts
- [ ] Multi-room — one phone per room, central alert hub
- [ ] Offline mode — on-device-only classification, no internet needed

---

## Priority Order

| Phase | Effort | Impact | When |
|-------|--------|--------|------|
| Phase 1 | 1-2 sessions | Makes it trustworthy | NOW |
| Phase 2 | 3-4 sessions | Biggest demo upgrade | Next |
| Phase 3 | 2-3 sessions | Engineering depth | After Phase 2 |
| Phase 4 | 5+ sessions | Product maturity | When time allows |
| Phase 5 | Long-term | Startup territory | After getting a job |

---

## Branch Strategy

- `main` — live deployed version (don't break this)
- `dev` — active development (Phase 1 + 2 work happens here)
- Merge `dev` → `main` only when a phase is complete and tested
