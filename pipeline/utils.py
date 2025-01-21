import re

def clean_vietnamese_text(text):
    """
    Cleans Vietnamese text by converting punctuation marks into spaces,
    converting the text to lowercase, and reducing multiple spaces to a single space.

    Args:
        text (str): The input Vietnamese text.

    Returns:
        str: Cleaned Vietnamese text.
    """
    # remove unk tokens
    text = text.replace("\x00", "")
    # Replace special characters and punctuation marks with a space
    text = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Reduce multiple spaces to a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespaces
    text = text.strip()

    return text

