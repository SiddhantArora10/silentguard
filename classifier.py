
# SilentGuard — Sound Classifier
# Maps YAMNet labels into alert categories and decides when to send a notification.

from notifier import send_alert

# --- Mode Definitions ---
# Each mode has a different set of sounds to watch for,
# depending on what Siddhant is doing at the time.

MODES = {
    "Sleep": {
        "description": "Hearing aids off — maximum coverage",
        "critical": [
            "Smoke detector, smoke alarm",
            "Fire alarm",
            "Alarm",
            "Carbon monoxide detector",
            "Glass",         # glass breaking
            "Shatter",
            "Baby cry",
            "Crying, sobbing",
            "Infant cry",
        ],
        "knock": [
            "Knock",
            "Doorbell",
            "Bang",
            "Slam",
            "Thump, thud",
            "Door",
            "Tap",
        ],
    },

    "Focus": {
        "description": "Working or studying — critical alerts only",
        "critical": [
            "Smoke detector, smoke alarm",
            "Fire alarm",
            "Alarm",
            "Carbon monoxide detector",
        ],
        "knock": [],  # no social sounds during focus time
    },

    "Music": {
        "description": "Listening to music — urgent + someone at the door",
        "critical": [
            "Smoke detector, smoke alarm",
            "Fire alarm",
            "Alarm",
            "Carbon monoxide detector",
        ],
        "knock": [
            "Knock",
            "Doorbell",
            "Bang",
            "Door",
        ],
    },
}

# How many consecutive detections before we alert for knock-type sounds
KNOCK_THRESHOLD = 2
MIN_CONFIDENCE = 0.40  # ignore anything below 40% confidence

# --- State ---
# Tracks how many times in a row we've heard a knock-type sound
consecutive_knock_count = 0


def classify(label, confidence, mode="Sleep"):
    """
    Takes a YAMNet label, confidence score, and the current listening mode.
    Decides whether to send an alert based on what sounds matter in that mode.
    Returns True if an alert was sent, False otherwise.
    """
    global consecutive_knock_count

    if confidence < MIN_CONFIDENCE:
        print(f"[low confidence] {label} ({confidence:.0%}) — below threshold")
        return False

    # Get the sound lists for the current mode
    critical_sounds = MODES[mode]["critical"]
    knock_sounds = MODES[mode]["knock"]

    # Check if it's a critical sound — alert immediately
    for sound in critical_sounds:
        if sound.lower() in label.lower():
            print(f"[CRITICAL] {label} ({confidence:.0%}) — alerting now")
            send_alert(f"[{mode} mode] {label}", confidence)
            consecutive_knock_count = 0
            return True

    # Check if it's a knock-type sound
    for sound in knock_sounds:
        if sound.lower() in label.lower():
            consecutive_knock_count += 1
            print(f"[KNOCK] {label} ({confidence:.0%}) — count: {consecutive_knock_count}/{KNOCK_THRESHOLD}")

            if consecutive_knock_count >= KNOCK_THRESHOLD:
                send_alert(f"[{mode} mode] {label}", confidence)
                consecutive_knock_count = 0
                return True
            return False

    # Not a sound we care about in this mode — reset and ignore
    consecutive_knock_count = 0
    print(f"[ambient] {label} ({confidence:.0%})")
    return False
