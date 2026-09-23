import re

import requests
from markdownify import markdownify
from requests.exceptions import RequestException


def visit_webpage(url: str, timeout: int = 30) -> str:
    """指定URLの内容を取得し、Markdown形式のテキストとして返す。

    Args:
        url: 取得対象のURL。
        timeout: リクエストのタイムアウト秒数。
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return str(markdown_content)
    except RequestException as e:
        return f"Error fetching the webpage: {e!s}"
    except Exception as e:
        return f"An unexpected error occurred: {e!s}"
