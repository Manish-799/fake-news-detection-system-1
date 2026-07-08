from backend.text_utils import clean_text


def test_clean_text_converts_to_lowercase():
    text = "President Announces NEW Policy"

    result = clean_text(text)

    assert result == "president announces new policy"


def test_clean_text_removes_urls():
    text = "Read more at https://example.com about the government"

    result = clean_text(text)

    assert result == "read more at about the government"


def test_clean_text_removes_html_tags():
    text = "<p>Government announces policy</p>"

    result = clean_text(text)

    assert result == "government announces policy"


def test_clean_text_removes_reuters_dateline():
    text = "WASHINGTON (Reuters) - President announces new policy."

    result = clean_text(text)

    assert result == "president announces new policy."


def test_clean_text_removes_news_agency_references():
    text = "Reuters reported the government announcement"

    result = clean_text(text)

    assert result == "reported the government announcement"


def test_clean_text_normalizes_whitespace():
    text = "Government     announces\n\nnew     policy"

    result = clean_text(text)

    assert result == "government announces new policy"


def test_clean_text_removes_unsupported_characters():
    text = "Government announces policy @#$ today"

    result = clean_text(text)

    assert result == "government announces policy today"