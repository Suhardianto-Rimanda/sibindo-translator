from app.services.nlp_processor import NlpProcessor


def test_smoothing_requires_consecutive_predictions():
    nlp = NlpProcessor(smoothing_window=3)
    assert not nlp.add_word("halo")
    assert not nlp.add_word("halo")
    # third matching consecutive prediction triggers acceptance
    assert nlp.add_word("halo")
    assert nlp.words() == ["halo"]


def test_dedup_skips_repeated_word():
    nlp = NlpProcessor(smoothing_window=2)
    nlp.add_word("halo")
    nlp.add_word("halo")  # accepted
    # next batch of "halo" should be deduped
    nlp.add_word("halo")
    nlp.add_word("halo")
    assert nlp.words() == ["halo"]


def test_stopword_ignored():
    nlp = NlpProcessor(smoothing_window=1)
    assert not nlp.add_word("__unknown__")
    assert nlp.words() == []


def test_sentence_capitalization():
    nlp = NlpProcessor(smoothing_window=1)
    nlp.add_word("halo")
    nlp.add_word("apa")
    nlp.add_word("kabar")
    assert nlp.sentence() == "Halo apa kabar"


def test_max_words_window():
    nlp = NlpProcessor(smoothing_window=1, max_words=2)
    nlp.add_word("a")
    nlp.add_word("b")
    nlp.add_word("c")
    assert nlp.words() == ["b", "c"]


def test_reset_clears_state():
    nlp = NlpProcessor(smoothing_window=1)
    nlp.add_word("halo")
    nlp.reset()
    assert nlp.words() == []
    assert nlp.sentence() == ""
