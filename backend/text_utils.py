import re


def clean_text(text: str) -> str:
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Remove Reuters-style datelines at the beginning
    #
    # Examples:
    # WASHINGTON (Reuters) -
    # LONDON (Reuters) -
    # SEATTLE/WASHINGTON (Reuters) -
    # WEST PALM BEACH, Fla./WASHINGTON (Reuters) -
    text = re.sub(
        r"^[a-z\s,\.\/\-]+"
        r"\(\s*reuters\s*\)\s*[-–—]\s*",
        " ",
        text,
        count=1,
    )

    # Remove remaining news agency references
    text = re.sub(
        r"\b("
        r"reuters|"
        r"associated press|"
        r"ap news|"
        r"afp"
        r")\b",
        " ",
        text,
    )

    # Remove image/media attribution artifacts
    text = re.sub(
        r"\b("
        r"featured image|"
        r"getty images|"
        r"getty image|"
        r"image credit|"
        r"photo credit|"
        r"photo courtesy|"
        r"stock photo"
        r")\b",
        " ",
        text,
    )

    # Remove standalone website media markers
    text = re.sub(
        r"\b("
        r"featured|"
        r"image|"
        r"images|"
        r"video"
        r")\b",
        " ",
        text,
    )

    # Keep alphanumeric characters and sentence punctuation
    text = re.sub(
        r"[^a-z0-9\s\.\,\!\?]",
        " ",
        text,
    )

    # Remove repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()