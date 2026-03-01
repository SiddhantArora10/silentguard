import os
import requests
from dotenv import load_dotenv

# Load secrets from .env file into environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_alert(sound_name, confidence):
    """Send a Telegram notification when a critical sound is detected."""

    message = f"🔔 SilentGuard Alert!\nSound detected: {sound_name}\nConfidence: {confidence:.0%}"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    response = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    })

    if response.status_code == 200:
        print(f"Alert sent: {sound_name}")
    else:
        print(f"Failed to send alert. Error: {response.text}")


# Test it directly when you run this file
if __name__ == "__main__":
    send_alert("Doorbell", 0.95)
