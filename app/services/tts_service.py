import os
import re
import uuid


class TtsService:
    """Convert text -> mp3 with gTTS. Returns a URL relative to /static."""

    def __init__(self, lang: str = "id", output_dir: str = "app/static/audio"):
        self.lang = lang
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def synthesize(self, text: str):
        if not text or not text.strip():
            return None

        try:
            from gtts import gTTS
        except ImportError:
            print("[TtsService] gTTS not installed")
            return None

        slug = re.sub(r"[^a-z0-9]+", "_", text.lower())[:24] or "tts"
        filename = f"{slug}_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(self.output_dir, filename)

        try:
            tts = gTTS(text=text, lang=self.lang)
            tts.save(filepath)
        except Exception as exc:
            print(f"[TtsService] synth failed: {exc}")
            return None

        return f"/static/audio/{filename}"
