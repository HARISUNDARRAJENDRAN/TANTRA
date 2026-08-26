from tantra_ml.text.vocabulary import Vocabulary, build_vocabulary, normalize_text


def test_multiscript_round_trip(tmp_path):
    texts = ["नमस्ते दुनिया", "வணக்கம் உலகம்", "হ্যালো বিশ্ব", "hello world"]
    vocabulary = build_vocabulary(texts)
    path = tmp_path / "vocab.json"
    vocabulary.save(path)
    loaded = Vocabulary.load(path)
    assert [loaded.decode(loaded.encode(text)) for text in texts] == texts
    assert loaded.tokens[0] == "<blank>"


def test_normalization():
    assert normalize_text("  hello\n world ") == "hello world"
