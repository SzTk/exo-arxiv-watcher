from unittest.mock import Mock, patch

import requests

from tools.web_fetch import visit_webpage


def test_visit_webpage_converts_html_to_markdown_text():
    fake_response = Mock()
    fake_response.text = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
    fake_response.raise_for_status = Mock()

    with patch("tools.web_fetch.requests.get", return_value=fake_response) as mock_get:
        result = visit_webpage("https://arxiv.org/abs/2401.00001")

    mock_get.assert_called_once()
    assert "Title" in result
    assert "Hello world" in result


def test_visit_webpage_returns_error_string_on_request_exception():
    with patch("tools.web_fetch.requests.get", side_effect=requests.exceptions.ConnectionError("boom")):
        result = visit_webpage("https://example.invalid/nope")

    assert result.startswith("Error fetching the webpage:")


def test_visit_webpage_returns_error_string_on_http_error():
    fake_response = Mock()
    fake_response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError("404"))

    with patch("tools.web_fetch.requests.get", return_value=fake_response):
        result = visit_webpage("https://arxiv.org/abs/nonexistent")

    assert result.startswith("Error fetching the webpage:")
