from threading import Lock


STOPWORDS = {"__unknown__", ""}


class NlpProcessor:
    """Rule-based NLP: dedup + smoothing + sentence assembly.

    - Dedup: skip if predicted word == last accepted word.
    - Smoothing: require N consecutive matching predictions before accepting.
    - Sentence assembly: append accepted words with a space, capitalize first.
    """

    def __init__(self, smoothing_window: int = 3, max_words: int = 30):
        self.smoothing_window = smoothing_window
        self.max_words = max_words
        self._words = []
        self._candidate = None
        self._candidate_count = 0
        self._lock = Lock()

    def add_word(self, word: str) -> bool:
        """Feed a predicted word into the smoother. Returns True if accepted."""
        if word in STOPWORDS:
            return False

        with self._lock:
            if word == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = word
                self._candidate_count = 1

            if self._candidate_count < self.smoothing_window:
                return False

            if self._words and self._words[-1] == word:
                # dedup blocked — clear candidate so a held gesture does not
                # inflate the counter and a new word still passes smoothing
                self._candidate = None
                self._candidate_count = 0
                return False

            self._words.append(word)
            if len(self._words) > self.max_words:
                self._words.pop(0)

            self._candidate = None
            self._candidate_count = 0
            return True

    def sentence(self) -> str:
        with self._lock:
            if not self._words:
                return ""
            text = " ".join(self._words)
            return text[0].upper() + text[1:]

    def words(self) -> list:
        with self._lock:
            return list(self._words)

    def reset(self):
        with self._lock:
            self._words.clear()
            self._candidate = None
            self._candidate_count = 0
