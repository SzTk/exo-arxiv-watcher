# Stage4 Augmented LLM (Tool Use) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `stage4_augmented.py` that extends Stage3's router→summarizer→tagger/evaluator pipeline with two tools — `find_similar_cases` (Tagger, self-accumulated local history) and `visit_webpage` (Evaluator, external knowledge) — plus a plain-Python `record_case` history writer invoked from code (not by the LLM).

**Architecture:** Stage0–3 files stay untouched. Two tool modules live under `tools/`: `tools/history_store.py` (JSONL-backed case history + TF-IDF similarity search) and `tools/web_fetch.py` (requests + markdownify page fetch, no API key required). `stage4_augmented.py` copies Stage3's structure, wires `tools=[find_similar_cases]` into the Tagger's `AgentConfig` and `tools=[visit_webpage]` into the Evaluator's, adds one instruction sentence to each, calls `record_case(...)` directly from `tag_with_evaluation` right after an "OK" verdict, and registers a small custom `any_agent.callbacks.base.Callback` that prints each tool call (name/args/output) to stdout for the demo log.

**Tech Stack:** Python 3.14, `any-agent[all]` / `any-llm-sdk[all]` (tinyagent backend), `requests` (already present), new deps `markdownify` and `scikit-learn`, `pytest` for the pure-function tests.

**Spec:** `stage4-design-doc.md` (repo root) — this plan implements it in full; executors should read both.

## Global Constraints

- Do not modify `stage0_fetch.py`, `stage1_chain.py`, `stage2_evaluator_optimizer.py`, or `stage3_routing.py` — Stage3 must remain a clean routing-only reference.
- New pipeline file is `stage4_augmented.py` at repo root, built from Stage3's structure.
- Tool implementations live under `tools/` (`tools/history_store.py`, `tools/web_fetch.py`).
- `record_case` is a plain Python function called directly from `tag_with_evaluation` after an "OK" verdict — it is never exposed to the LLM as a tool.
- `find_similar_cases` is exposed only on the Tagger's `AgentConfig(tools=[...])`; `visit_webpage` is exposed only on the Evaluator's.
- `TAGGER_INSTRUCTIONS` gets exactly one appended sentence: "候補タグの選定に迷う場合は、find_similar_casesツールで類似した過去の確定事例を参照し、判断の参考にしてよい。"
- `EVALUATOR_INSTRUCTIONS` gets exactly one appended sentence: "abstractの記述だけでは判断が難しい場合、visit_webpageツールで論文ページ(paper_idのURL)を直接確認してよい。" — this requires the evaluator prompt to include `paper_id` (the arXiv abstract-page URL already present as `paper["id"]` from `stage0_fetch.py`).
- History file path is `data/tag_history.jsonl`, created on first write; `data/` must be added to `.gitignore`.
- Prefer web/similarity implementations that need no extra API key (arXiv URLs are directly fetchable; TF-IDF via scikit-learn needs no external service).

## Review Focus

- `find_similar_cases` called before `data/tag_history.jsonl` exists (very first run) must return `[]`, not raise `FileNotFoundError`.
- `find_similar_cases` with fewer than `top_k` history records, or where every similarity score is 0 (unrelated abstracts), must not crash and must drop zero-similarity entries per the design doc's `if s > 0` filter.
- `visit_webpage` called with an unreachable URL or a non-2xx response must return a readable error string (e.g. `"Error fetching the webpage: ..."`), not raise, since an uncaught exception inside a tool call would break the agent's tool-calling loop.
- `record_case` writes must round-trip abstracts containing newlines, quotes, and non-ASCII (Japanese tags) correctly through `json.dumps(..., ensure_ascii=False)` / `json.loads`.
- `tag_with_evaluation` must call `record_case` only on the attempt where the verdict is "OK" — if `MAX_RETRIES` is exhausted without an OK verdict, `record_case` must NOT be called for that paper.

---

## Task 1: `tools/history_store.py` — case history + similarity search

**Files:**
- Create: `tools/__init__.py` (empty)
- Create: `tools/history_store.py`
- Create: `tests/test_history_store.py`
- Modify: `pyproject.toml` (add `scikit-learn` and `pytest` dependencies)
- Modify: `.gitignore` (add `data/`)

**Interfaces:**
- Produces: `record_case(paper_id: str, title: str, abstract: str, tags: str) -> None` — appends one JSON line to `HISTORY_PATH`.
- Produces: `find_similar_cases(abstract: str, top_k: int = 3) -> list[dict]` — each dict has keys `title`, `tags`, `similarity` (float, rounded to 3 decimals); results sorted by similarity descending, filtered to `similarity > 0`, capped at `top_k`.
- Produces: `HISTORY_PATH: Path` — importable so tests can monkeypatch it.

- [ ] **Step 1: Add dependencies**

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
uv add scikit-learn
uv add --dev pytest
```

- [ ] **Step 2: Add `data/` to `.gitignore`**

Append to `.gitignore`:

```
# Stage4 runtime data
data/
```

- [ ] **Step 3: Create `tools/__init__.py`**

Empty file (makes `tools` an explicit package so `tests/` can `from tools.history_store import ...` unambiguously).

- [ ] **Step 4: Write the failing tests**

Create `tests/test_history_store.py`:

```python
import json

import pytest

from tools import history_store


@pytest.fixture(autouse=True)
def isolated_history_path(tmp_path, monkeypatch):
    path = tmp_path / "tag_history.jsonl"
    monkeypatch.setattr(history_store, "HISTORY_PATH", path)
    return path


def test_find_similar_cases_returns_empty_when_file_missing():
    assert history_store.find_similar_cases("some abstract about exoplanets") == []


def test_record_case_then_find_similar_cases_ranks_by_similarity():
    history_store.record_case(
        "id1", "Transit photometry survey", "We detect exoplanets using transit photometry and light curves.", "検出手法"
    )
    history_store.record_case(
        "id2", "DMS biosignature study", "We study DMS and O2 biosignatures in exoplanet atmospheres.", "バイオシグネチャ"
    )
    history_store.record_case(
        "id3", "Orbital dynamics of binaries", "We model orbital dynamics and resonances of binary star systems.", "軌道力学"
    )

    results = history_store.find_similar_cases(
        "This paper reports transit photometry detection of a new exoplanet via light curve analysis.",
        top_k=2,
    )

    assert len(results) <= 2
    assert results[0]["title"] == "Transit photometry survey"
    assert results[0]["tags"] == "検出手法"
    assert 0.0 < results[0]["similarity"] <= 1.0
    assert list(results[i]["similarity"] for i in range(len(results))) == sorted(
        (r["similarity"] for r in results), reverse=True
    )


def test_record_case_round_trips_unicode_and_newlines(isolated_history_path):
    history_store.record_case(
        "id4", "タイトル\n改行あり", 'abstract with "quotes" and\nnewlines', "大気科学, その他"
    )
    lines = isolated_history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["title"] == "タイトル\n改行あり"
    assert record["tags"] == "大気科学, その他"
    assert "quotes" in record["abstract"]


def test_find_similar_cases_drops_zero_similarity_entries():
    history_store.record_case("id5", "Unrelated topic", "zzz completely unrelated qqq wibble", "その他")
    results = history_store.find_similar_cases("transit photometry exoplanet detection light curve", top_k=3)
    assert results == []
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && uv run pytest tests/test_history_store.py -v`
Expected: FAIL / ERROR — `tools.history_store` does not exist yet (`ModuleNotFoundError`).

- [ ] **Step 6: Implement `tools/history_store.py`**

```python
import json
from pathlib import Path

HISTORY_PATH = Path("data/tag_history.jsonl")


def record_case(paper_id: str, title: str, abstract: str, tags: str) -> None:
    """確定したタグ付け結果を履歴に追記する。

    LLMツールではなく、evaluatorがOKを出した後にコードから直接呼ぶ。
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        record = {"id": paper_id, "title": title, "abstract": abstract, "tags": tags}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_similar_cases(abstract: str, top_k: int = 3) -> list[dict]:
    """与えられたabstractに類似した、過去に確定したタグ付け事例を返す。

    Args:
        abstract: 類似度を計算する対象のabstract(英語)。
        top_k: 返す件数の上限。

    Taggerがタグ選定に迷った際に呼び出すツールとして公開する。
    """
    if not HISTORY_PATH.exists():
        return []
    lines = [l for l in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    if not records:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [r["abstract"] for r in records] + [abstract]
    vectors = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    ranked = sorted(zip(sims, records), key=lambda x: x[0], reverse=True)
    return [
        {"title": r["title"], "tags": r["tags"], "similarity": round(float(s), 3)}
        for s, r in ranked[:top_k]
        if s > 0
    ]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && uv run pytest tests/test_history_store.py -v`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
git add tools/__init__.py tools/history_store.py tests/test_history_store.py pyproject.toml uv.lock .gitignore
git commit -m "feat: add tag history store with TF-IDF similarity search"
```

---

## Task 2: `tools/web_fetch.py` — `visit_webpage` tool

**Files:**
- Create: `tools/web_fetch.py`
- Create: `tests/test_web_fetch.py`
- Modify: `pyproject.toml` (add `markdownify` dependency)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `visit_webpage(url: str, timeout: int = 30) -> str` — fetches `url`, converts HTML to markdown text, collapses repeated blank lines; on any request failure returns a string starting with `"Error fetching the webpage: "` (never raises).

- [ ] **Step 1: Add dependency**

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
uv add markdownify
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_web_fetch.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && uv run pytest tests/test_web_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.web_fetch'`.

- [ ] **Step 4: Implement `tools/web_fetch.py`**

```python
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
```

Note: this is adapted from the reference `agent_webpage.py` implementation — same signature and error-handling shape, requires no external API key. The regex is tightened from the reference (`\n{2,}` → `\n{3,}` collapsing to a single blank line) so paragraph breaks survive; either is acceptable, but keep the "≥2 blank lines → 1 blank line" behavior your test asserts.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && uv run pytest tests/test_web_fetch.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Manual sanity check against a real URL**

Run:

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
uv run python -c "from tools.web_fetch import visit_webpage; print(visit_webpage('https://arxiv.org/abs/2401.00001')[:500])"
```

Expected: prints readable markdown text from the arXiv abstract page (title/abstract text visible), not an error string.

- [ ] **Step 7: Commit**

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
git add tools/web_fetch.py tests/test_web_fetch.py pyproject.toml uv.lock
git commit -m "feat: add visit_webpage tool for fetching paper pages as markdown"
```

---

## Task 3: `stage4_augmented.py` — wire tools into the Stage3 pipeline

**Files:**
- Create: `stage4_augmented.py`

**Interfaces:**
- Consumes: `tools.history_store.record_case(paper_id: str, title: str, abstract: str, tags: str) -> None`, `tools.history_store.find_similar_cases(abstract: str, top_k: int = 3) -> list[dict]`, `tools.web_fetch.visit_webpage(url: str, timeout: int = 30) -> str`, `stage0_fetch.fetch_new_papers(category="astro-ph.EP", max_results=20) -> list[dict]` (each paper dict has `id`, `title`, `summary`, `published`, per `stage0_fetch.py`).
- Produces: `tag_with_evaluation(paper_id: str, title: str, abstract: str, jp_summary: str, tagger, evaluator) -> tuple[str, list[dict]]` (note the two new leading params vs. Stage3's version — this file does not modify `stage3_routing.py`, it defines its own copy).

- [ ] **Step 1: Write `stage4_augmented.py`**

```python
import json
import os

from any_agent import AgentConfig, AnyAgent
from any_agent.callbacks.base import Callback
from any_agent.tracing.attributes import GenAI

from tools.history_store import find_similar_cases, record_case
from tools.web_fetch import visit_webpage

MODEL_ID = os.environ.get("MODEL_ID", "openai:Qwen2.5-7B-Instruct-Q8_0")
API_BASE = os.environ.get("API_BASE", "http://localhost:8080/v1")
API_KEY = os.environ.get("API_KEY", "whatever")

TAG_LIST = ["検出手法", "バイオシグネチャ", "大気科学", "軌道力学", "観測装置", "その他"]
MAX_RETRIES = 2

ROUTER_INSTRUCTIONS = (
    "あなたは論文の一次選別担当です。与えられた論文タイトルとabstract(英語)を読み、"
    "次のどちらかに強く関連するかを判定してください。\n"
    "(a) 系外惑星の検出手法(トランジット法、視線速度法/ドップラー法、アストロメトリ、直接撮像など)\n"
    "(b) 系外惑星大気のバイオシグネチャ(O2, CH4, DMS/DMDSなど)や大気組成・habitability\n"
    "どちらかに強く関連するなら「RELEVANT」とだけ出力してください。"
    "彗星、太陽系内天体、惑星形成の一般理論、観測装置そのものの技術論文など、"
    "(a)(b)に直接関係しないものは「SKIP: <理由>」の形式で出力してください。"
    "余計な文章は含めないでください。"
)

SUMMARIZE_INSTRUCTIONS = (
    "あなたは天文学論文の要約者です。"
    "与えられた英語のabstractを、日本語で3行以内に要約してください。"
    "専門用語は残してよいですが、平易な説明を心がけてください。"
    "要約以外の文章は出力しないでください。"
)

TAGGER_INSTRUCTIONS = (
    "あなたは論文分類者です。与えられた要約(と、あれば直前の査読コメント)を読み、次のタグ候補から"
    f"最も当てはまるものを1〜2個選んでください: {', '.join(TAG_LIST)}。"
    "出力はカンマ区切りのタグ名のみにし、他の文章は含めないでください。"
    "候補タグの選定に迷う場合は、find_similar_casesツールで類似した過去の確定事例を参照し、"
    "判断の参考にしてよい。"
)

EVALUATOR_INSTRUCTIONS = (
    "あなたはタグ付けの厳格な査読者です。次の手順で判定してください。\n"
    f"候補タグ: {', '.join(TAG_LIST)}\n"
    "1. 候補タグそれぞれについて、abstractの内容に当てはまるかを一言で検討する"
    "(例: 検出手法: ○/× 理由、バイオシグネチャ: ○/× 理由、…を全候補について行う)\n"
    "2. 1の検討結果に基づき、最も当てはまる1〜2個のタグを選ぶ\n"
    "3. 2で選んだタグと、実際に付けられたタグを比較する\n"
    "4. 一致していれば出力の最後の行に「OK」とだけ書く。"
    "一致していなければ最後の行に「NG: <理由と正しいタグ>」と書く\n"
    "1〜3の検討過程は出力してよいが、必ず最後の1行を4の形式にすること。"
    "abstractの記述だけでは判断が難しい場合、visit_webpageツールで論文ページ(paper_idのURL)を"
    "直接確認してよい。"
)


class ToolCallLogger(Callback):
    """ツール呼び出しをデモ資料用に標準出力へ記録する薄いラッパー。"""

    def after_tool_execution(self, context, *args, **kwargs):
        span = context.current_span
        if span.attributes.get(GenAI.OPERATION_NAME) != "execute_tool":
            return context

        tool_name = span.attributes.get(GenAI.TOOL_NAME)
        tool_args = span.attributes.get(GenAI.TOOL_ARGS, "{}")
        output = span.attributes.get(GenAI.OUTPUT, "")
        output_preview = output if len(output) <= 200 else output[:200] + "..."
        print(f"  [tool call] {tool_name}({tool_args}) -> {output_preview}")
        return context


def build_agent(instructions: str, tools: list | None = None) -> AnyAgent:
    return AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id=MODEL_ID,
            api_key=API_KEY,
            api_base=API_BASE,
            instructions=instructions,
            tools=tools or [],
            callbacks=[ToolCallLogger()],
        ),
    )


def tag_with_evaluation(paper_id: str, title: str, abstract: str, jp_summary: str, tagger, evaluator):
    log = []
    feedback = ""
    tags = ""
    for attempt in range(1, MAX_RETRIES + 2):
        prompt = jp_summary if not feedback else f"{jp_summary}\n\n前回の査読コメント: {feedback}\nこれを踏まえて選び直してください。"
        tags = tagger.run(prompt).final_output.strip()

        eval_prompt = f"paper_id (URL): {paper_id}\nabstract: {abstract}\n\n付けられたタグ: {tags}"
        verdict_full = evaluator.run(eval_prompt).final_output.strip()
        verdict_last_line = verdict_full.strip().splitlines()[-1].strip()

        log.append({"attempt": attempt, "tags": tags, "verdict": verdict_full})
        if verdict_last_line.upper().startswith("OK"):
            record_case(paper_id, title, abstract, tags)
            break
        feedback = verdict_last_line

    return tags, log


def main():
    from stage0_fetch import fetch_new_papers

    router = build_agent(ROUTER_INSTRUCTIONS)
    summarizer = build_agent(SUMMARIZE_INSTRUCTIONS)
    tagger = build_agent(TAGGER_INSTRUCTIONS, tools=[find_similar_cases])
    evaluator = build_agent(EVALUATOR_INSTRUCTIONS, tools=[visit_webpage])

    papers = fetch_new_papers(max_results=10)  # フィルタでかなり落ちる想定で少し多めに
    for p in papers:
        print(f"=== {p['title']} ===")

        route_prompt = f"title: {p['title']}\nabstract: {p['summary']}"
        verdict = router.run(route_prompt).final_output.strip()

        if not verdict.upper().startswith("RELEVANT"):
            print(f"[SKIP] {verdict}\n")
            continue

        jp_summary = summarizer.run(p["summary"]).final_output
        print(f"[要約] {jp_summary}")

        final_tags, log = tag_with_evaluation(p["id"], p["title"], p["summary"], jp_summary, tagger, evaluator)
        print(f"[最終タグ] {final_tags}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && uv run python -c "import stage4_augmented"`
Expected: no errors (imports resolve, no syntax errors). This does not contact an LLM server.

- [ ] **Step 3: Confirm `GenAI`/`Callback` import paths match the installed `any-agent` version**

Run:

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
uv run python -c "from any_agent.callbacks.base import Callback; from any_agent.tracing.attributes import GenAI; print('ok')"
```

Expected: prints `ok`. If either import fails, adjust `stage4_augmented.py`'s imports to whatever path this project's installed `any-agent` version exposes (check `uv run python -c "import any_agent; print(any_agent.__file__)"` and inspect `any_agent/callbacks/` and `any_agent/tracing/attributes.py` in that install) and re-run this step before continuing.

- [ ] **Step 4: Manual end-to-end run against a live local model (design doc's staged verification, steps 2-5)**

Requires a local OpenAI-compatible server reachable at `API_BASE` (e.g. llama.cpp server), matching how `stage3_routing.py` is normally run. With that server running:

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
rm -f data/tag_history.jsonl   # start from a clean history for the first observation
uv run python stage4_augmented.py
```

Expected:
- `data/tag_history.jsonl` is created and gains one line per paper that reaches an "OK" verdict.
- `[tool call] visit_webpage(...)` lines appear in stdout whenever the evaluator chooses to fetch a page.
- `[tool call] find_similar_cases(...)` lines appear whenever the tagger chooses to consult history (on the first run this will be `[]` since history starts empty; run the script a second time to see nonempty matches).
- No unhandled exceptions; a failed `visit_webpage` call (bad URL) surfaces as an "Error fetching the webpage: ..." string inside the tool-call log line, not a crash.

- [ ] **Step 5: Commit**

```bash
cd /home/taka/Documents/mygit/exo-arxiv-watcher
git add stage4_augmented.py
git commit -m "feat: add stage4 augmented LLM pipeline with tagger/evaluator tools"
```

---

## Task 4: Final check — Stage3 untouched, README pointer (optional)

**Files:**
- Verify only: `stage0_fetch.py`, `stage1_chain.py`, `stage2_evaluator_optimizer.py`, `stage3_routing.py`

- [ ] **Step 1: Confirm no earlier-stage files were modified**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && git status --short stage0_fetch.py stage1_chain.py stage2_evaluator_optimizer.py stage3_routing.py`
Expected: empty output (no changes staged or unstaged for these files).

- [ ] **Step 2: Full test suite run**

Run: `cd /home/taka/Documents/mygit/exo-arxiv-watcher && uv run pytest -v`
Expected: all tests from Task 1 and Task 2 pass; no other test files affected.
