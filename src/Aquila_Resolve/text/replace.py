import re


def replace_first(target: str, replacement: str, text: str) -> str:
    """
    Use Regex to replace the first instance of a word

    Words within braces are ignored (e.g. '{word} is ignored')

    :param target: The word to be replaced
    :param replacement: Replacement word
    :param text: Text to be searched
    :return: Text with the first instance of the word replaced
    """
    if not target or not text:
        return text  # Return original if no target or text
    # Replace the first instance of a word with its phonemes
    # return re.sub(r'(?i)\b' + target + r'\b', replacement, text, 1)
    return re.sub(r'(?<!\{)\b' + target + r'\b(?![\w\s]*[}])', replacement, text, count=1, flags=re.IGNORECASE)
