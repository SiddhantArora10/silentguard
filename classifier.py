
# SilentGuard — Sound Classifier
# Maps YAMNet labels into alert categories and decides when to send a notification.

from notifier import send_alert

# --- Sound Categories ---

# These trigger an alert IMMEDIATELY — no waiting
CRITICAL_SOUNDS = [
    "Smoke detector, smoke alarm",
    "Fire alarm",
    "Alarm",
    "Carbon monoxide detector",
]

# These trigger an alert only after detected 2+ times in a row
# (filters out random single bangs or accidental sounds)
KNOCK_SOUNDS = [
    "Knock",
    "Doorbell",
    "Bang",
    "Slam",
    "Thump, thud",
    "Door",
    "Tap",
]

# How many consecutive detections before we alert for knock-type sounds
KNOCK_THRESHOLD = 2
MIN_CONFIDENCE = 0.40  # Minimum confidence to consider a detection valid

# --- State ---
# Tracks how many times in a row we've heard a knock-type sound
consecutive_knock_count = 0


def classify(label, confidence):
    """
    Takes a YAMNet label and confidence score.
    Decides whether to send an alert.
    Returns True if an alert was sent, False otherwise.
    """
    global consecutive_knock_count

    if confidence < MIN_CONFIDENCE:
        print(f"[low confidence] {label} ({confidence:.0%}) — below threshold")
        return False
    
    # Check if it's a critical sound — alert immediately
    for sound in CRITICAL_SOUNDS:
        if sound.lower() in label.lower():
            print(f"[CRITICAL] {label} ({confidence:.0%}) — alerting now")
            send_alert(label, confidence)
            consecutive_knock_count = 0  # reset knock counter
            return True

    # Check if it's a knock-type sound
    for sound in KNOCK_SOUNDS:
        if sound.lower() in label.lower():
            consecutive_knock_count += 1
            print(f"[KNOCK] {label} ({confidence:.0%}) — count: {consecutive_knock_count}/{KNOCK_THRESHOLD}")

            if consecutive_knock_count >= KNOCK_THRESHOLD:
                send_alert(label, confidence)
                consecutive_knock_count = 0  # reset after alerting
                return True
            return False

    # Not a sound we care about — reset knock counter and ignore
    consecutive_knock_count = 0
    print(f"[ambient] {label} ({confidence:.0%})")
    return False
