import re
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

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

def convert_bold_to_html(text: str) -> str:
    """
    Converts Markdown-style **bold** text into HTML <b> tags
    by processing each ** one at a time and alternating between
    <b> and </b>.
    """
    result = []
    is_bold_open = False  # Tracks whether the next ** opens or closes a tag
    i = 0

    while i < len(text):
        if text[i:i+2] == "**":  # Found a bold marker
            # Add the appropriate tag
            if is_bold_open:
                result.append("</b>")
            else:
                result.append("<b>")
            is_bold_open = not is_bold_open  # Toggle the state
            i += 2  # Skip over the `**`
        else:
            result.append(text[i])  # Append the current character
            i += 1

    return "".join(result)

def markdown_to_custom_html(text):
    """Converts markdown text (bold, new lines) into custom HTML with <span> and <br> formatting."""
    
    # Convert **bold** to <span class="r1">...</span>
    text = re.sub(r"\*\*(.*?)\*\*", r'<span class="r1">\1</span>', text)
    
    # Convert \n to <br>
    text = text.replace("\n", "<br>")

    # Wrap in <code> tag
    return "<style>.r1 {font-weight: bold}</style>" + f'<code style="font-family:inherit">{text}</code>'

def duckduckgo_search(query: str, max_results: int = 5):
    """
    Perform a DuckDuckGo search and return structured results.
    
    Parameters:
        query (str): The search query.
        max_results (int): Number of results to retrieve.
        
    Returns:
        list[dict]: A list of dictionaries containing title, link, and snippet.
    """
    search_tool = DuckDuckGoSearchResults(output_format="list")
    results = search_tool.run(query, num_results=max_results)

    return results # list[dict(snippet, title, link)]