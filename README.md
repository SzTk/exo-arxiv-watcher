# exo-arxiv-watcher

arXivに日々投稿される系外惑星(exoplanet)関連論文を、LLMエージェントで自動的に選別・要約・分類するウォッチャー。

[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) で紹介されている
基本的なエージェント設計パターンを、実際の題材(arXiv astro-ph.EPカテゴリの論文トリアージ)で
一つずつ体験しながら実装していくための、学習用の段階的実装です。Stage0〜4まで、後のステージほど
前のステージにパターンを積み重ねていきます。

## パイプラインの構成

| ステージ | ファイル | 設計パターン | 内容 |
|---|---|---|---|
| Stage0 | [stage0_fetch.py](stage0_fetch.py) | - | arXiv APIから最新論文を取得するだけの下準備 |
| Stage1 | [stage1_chain.py](stage1_chain.py) | Prompt Chaining | 要約 → タグ付け、を直列に繋ぐ最小構成 |
| Stage2 | [stage2_evaluator_optimizer.py](stage2_evaluator_optimizer.py) | Evaluator-Optimizer | タグ付け結果をEvaluatorが査読し、NGなら理由を添えてTaggerに再試行させるループ |
| Stage3 | [stage3_routing.py](stage3_routing.py) | Routing | 本題(検出手法/バイオシグネチャ)に関連するかをRouterが一次選別し、無関係な論文はスキップ |
| Stage4 | [stage4_augmented.py](stage4_augmented.py) | Augmented LLM(ツール利用) | Taggerに過去の確定事例を参照する`find_similar_cases`、Evaluatorに論文ページを直接読む`visit_webpage`を付与 |

Stage4の設計意図は [stage4-design-doc.md](stage4-design-doc.md) に、全体のデータフローは
[ExoArXivPipeline_Diagram.md](ExoArXivPipeline_Diagram.md) にまとめています。

## セットアップ

Python 3.14以上、[uv](https://docs.astral.sh/uv/) が必要です。

```bash
uv sync
cp .env.example .env
```

LLMバックエンドは `.env` の `MODEL_ID` / `API_BASE` / `API_KEY` で切り替えます([tools/llm_config.py](tools/llm_config.py))。

- デフォルトはローカルのllamafile/llama.cpp(OpenAI互換API、`http://localhost:8080/v1`)を想定
- Anthropic APIやOpenAI互換のリレー経由でクラウドモデルを使う場合の設定例は `.env.example` を参照

## 実行方法

各ステージは単体のスクリプトとして実行できます。

```bash
uv run python stage0_fetch.py
uv run python stage1_chain.py
uv run python stage2_evaluator_optimizer.py
uv run python stage3_routing.py
uv run python stage4_augmented.py
```

Stage4はTagger/Evaluatorの判断過程やツール呼び出しのログを標準出力に表示します。また、確定したタグ付け結果は
`data/tag_history.jsonl` に蓄積され、以降の`find_similar_cases`による類似事例参照に使われます。

## テスト

```bash
uv run pytest
```

## 補足

- `stage0_fetch.py` はWSL2環境で`curl`がarXiv API(export.arxiv.org)からTLSフィンガープリントの都合で
  弾かれる問題を回避するため、WSL2上ではWindows側の`curl.exe`を優先的に使用します。
