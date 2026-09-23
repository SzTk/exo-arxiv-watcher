# Stage4設計: ツール拡張(Augmented LLM)実装計画

## 背景・目的

exo-arxiv-watcherはこれまでStage1(prompt chaining)、Stage2(evaluator-optimizer)、Stage3(routing)を実装済み。
いずれのサブエージェントもinstructionが異なるだけでtools/modelは共通(tools=[])だった。

今回、Building Effective Agentsの基本構成要素である「Augmented LLM(ツール利用)」を体験する題材として、
既存のTagger/Evaluatorに2種類のツールを追加する。

- ツールA: `visit_webpage` — Evaluatorに、abstract以外の情報(論文ページ本文)を参照させ、判定の根拠を広げる。外部知識で補うアプローチ。
- ツールB: `find_similar_cases` — Taggerに、過去に確定したタグ付け事例をローカルに蓄積し、迷ったときに参照させる。自己蓄積データで補うアプローチ。

この2つは「ツールが与える知識の性質の違い(外部 vs 自己蓄積)」を対比できる組み合わせとして選んだ。

## 全体方針

- Stage3までのファイルは変更せず、`stage4_augmented.py`を新規作成してStage3の構成をベースに拡張する(seminar資料としてStage3を「routingのみのクリーンな例」として残すため)。
- ツール本体は`tools/`以下に切り出し、any-agentのAgentConfigへ`tools=[...]`として渡す。
- 履歴の記録(`record_case`)はLLMが判断して呼ぶツールではなく、evaluatorがOKを出した後にコード側から直接呼ぶ通常のPython関数とする(記録するかどうかはロジックで決まっており、LLMの判断を挟む理由がないため)。
- 類似事例の検索(`find_similar_cases`)は逆に、Tagger自身が「迷ったら呼ぶ」と判断するツールとして公開する。

## ファイル構成案

```
exo-arxiv-watcher/
  stage0_fetch.py                 (既存、変更なし)
  stage1_chain.py                 (既存、変更なし)
  stage2_evaluator_optimizer.py   (既存、変更なし)
  stage3_routing.py               (既存、変更なし)
  stage4_augmented.py             (新規)
  tools/
    history_store.py              (新規)
    web_fetch.py                  (新規、既存のvisit_webpage実装をラップ)
  data/
    tag_history.jsonl             (実行時に生成。.gitignore対象)
```

## 各コンポーネントの詳細設計

### tools/history_store.py

```python
import json
from pathlib import Path

HISTORY_PATH = Path("data/tag_history.jsonl")


def record_case(paper_id: str, title: str, abstract: str, tags: str) -> None:
    """確定したタグ付け結果を履歴に追記する。
    LLMツールではなく、evaluatorがOKを出した後にコードから直接呼ぶ。
    """
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        record = {"id": paper_id, "title": title, "abstract": abstract, "tags": tags}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_similar_cases(abstract: str, top_k: int = 3) -> list[dict]:
    """与えられたabstractに類似した、過去に確定したタグ付け事例を返す。
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

備考: 類似度計算はTF-IDF+コサイン類似度(scikit-learn)を想定。依存を増やしたくない場合は`difflib.SequenceMatcher`ベースの簡易版でも代替可(精度は落ちるがゼロ依存)。どちらを採るかはClaude Code側で判断してよい。

### tools/web_fetch.py

```python
"""
既存のvisit_webpage実装(ddgs/Brave/SearXNG系、またはany-agent組み込みのもの)を
そのまま流用する想定のラッパー。

今回はarXiv側のURL(stage0_fetchで取得済みのpaper["id"])が既知なので、
検索(search_web)は不要で、visit_webpageだけあれば足りる。

想定シグネチャ:
    def visit_webpage(url: str) -> str:
        '''指定URLの内容をテキストとして返す'''
        ...

Claude Codeへの依頼:
    手元のexample集にある実装をここにコピーまたはimportし、
    上記シグネチャに合わせて薄くラップしてください。
    採用する実装は、追加の外部API keyが不要なものを優先してください。
"""
```

### stage4_augmented.py の変更点(stage3_routing.pyとの差分)

- 全体の流れ(router → summarizer → tagger → evaluatorループ)はstage3のまま。
- Taggerの`AgentConfig`に`tools=[find_similar_cases]`を追加。
- Evaluatorの`AgentConfig`に`tools=[visit_webpage]`を追加。
- `TAGGER_INSTRUCTIONS`に一文追記:
  「候補タグの選定に迷う場合は、find_similar_casesツールで類似した過去の確定事例を参照し、判断の参考にしてよい。」
- `EVALUATOR_INSTRUCTIONS`に一文追記:
  「abstractの記述だけでは判断が難しい場合、visit_webpageツールで論文ページ(paper_idのURL)を直接確認してよい。」
- `tag_with_evaluation`ループ内、verdictが"OK"になった直後に`record_case(paper_id, title, abstract, tags)`を呼ぶ(通常のPython関数呼び出しとして)。

## 実装・検証の進め方(段階的に確認しながら)

1. `history_store.py`を単体で書き、手動で3〜4件のダミーJSONLを用意して`find_similar_cases`の類似度ランキングが妥当か確認する(agentを介さずに動作確認)。
2. stage3のフローに`record_case`だけを組み込み、1回通して`tag_history.jsonl`が正しく蓄積されるか確認する。
3. Taggerに`find_similar_cases`のみ追加し、既存のテスト用paper群で再実行。ツール未装備時(stage2/3のログ)と比較し、タグの変化・NGループ回数の変化を見る。
4. Evaluatorに`visit_webpage`のみ追加し、再実行。ツールが実際に呼ばれる頻度と、呼ばれた場合の判定の変化を観察する。
5. 両方を組み込んだ`stage4_augmented.py`をフルセットの新着論文で通し、ツール呼び出しログをデモ資料用に記録する。

## Claude Codeへの申し送り事項(要確認・要選択)

- 既存のvisit_webpage/search実装のうち、どれを採用するか(ddgs / Brave API / SearXNG / any-agent組み込み)。追加のAPI keyが不要なものを優先推奨。
- any-agentのtinyagentが素のPython関数(型ヒント+docstring)をどうツールスキーマに変換しているか、手元の別exampleコードのパターンに合わせて実装する。
- 類似度計算にscikit-learnを追加するか、依存を増やさずdifflibベースの簡易版にするか。
- ツール呼び出しをログ出力する薄いラッパーを挟むかどうか(SPARQLツールのpre/post callbackの発想を流用できる)。

## 依存関係の変更(想定)

- `uv add scikit-learn` (類似度計算用。difflib案を採る場合は不要)
- 採用するweb系ツール実装が要求する依存(ddgs, requests, beautifulsoup4等) — Claude Code側で確認の上追加