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
