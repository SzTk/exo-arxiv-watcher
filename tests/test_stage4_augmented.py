from types import SimpleNamespace
from unittest.mock import patch

import pytest

import stage4_augmented as s4


class StubResult:
    def __init__(self, final_output):
        self.final_output = final_output


class StubAgent:
    """Records every prompt it was called with and returns queued outputs in order."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def run(self, prompt):
        self.prompts.append(prompt)
        return StubResult(self.outputs.pop(0))


class RaisingAgent:
    """Simulates a tool-calling agent that hit the max-tool-calls guard."""

    def run(self, prompt):
        raise s4.MaxToolCallsExceeded("tool called 6 times (limit 5)")


def test_tag_with_evaluation_includes_abstract_in_tagger_prompt():
    tagger = StubAgent(["検出手法"])
    evaluator = StubAgent(["1. ...\nOK"])

    with patch.object(s4, "record_case") as mock_record_case:
        s4.tag_with_evaluation(
            "id1", "Title", "We detect exoplanets via transit photometry.", "日本語要約", tagger, evaluator
        )

    assert "We detect exoplanets via transit photometry." in tagger.prompts[0]
    mock_record_case.assert_called_once()


def test_tag_with_evaluation_calls_record_case_only_on_ok_verdict():
    tagger = StubAgent(["検出手法"])
    evaluator = StubAgent(["1. ...\nOK"])

    with patch.object(s4, "record_case") as mock_record_case:
        tags, log = s4.tag_with_evaluation("id1", "Title", "abstract text", "要約", tagger, evaluator)

    mock_record_case.assert_called_once_with("id1", "Title", "abstract text", "検出手法")
    assert tags == "検出手法"
    assert len(log) == 1


def test_tag_with_evaluation_does_not_call_record_case_when_retries_exhausted():
    tagger = StubAgent(["検出手法", "その他", "軌道力学"])
    evaluator = StubAgent(["NG: 違う", "NG: まだ違う", "NG: 最後まで違う"])

    with patch.object(s4, "record_case") as mock_record_case:
        tags, log = s4.tag_with_evaluation("id1", "Title", "abstract text", "要約", tagger, evaluator)

    mock_record_case.assert_not_called()
    assert len(log) == s4.MAX_RETRIES + 1


def test_tag_with_evaluation_recovers_from_max_tool_calls_without_crashing():
    tagger = RaisingAgent()
    evaluator = StubAgent(["should not be reached"])

    with patch.object(s4, "record_case") as mock_record_case:
        tags, log = s4.tag_with_evaluation("id1", "Title", "abstract text", "要約", tagger, evaluator)

    mock_record_case.assert_not_called()
    assert len(log) == 1
    assert "ツール呼び出し上限" in log[0]["verdict"]


def test_tool_call_logger_raises_after_limit_exceeded():
    logger = s4.ToolCallLogger(max_tool_calls=2)
    context = SimpleNamespace(shared={})

    logger.before_tool_execution(context)
    logger.before_tool_execution(context)
    with pytest.raises(s4.MaxToolCallsExceeded):
        logger.before_tool_execution(context)


def test_tool_call_logger_allows_calls_up_to_limit():
    logger = s4.ToolCallLogger(max_tool_calls=2)
    context = SimpleNamespace(shared={})

    logger.before_tool_execution(context)
    result = logger.before_tool_execution(context)

    assert result is context
