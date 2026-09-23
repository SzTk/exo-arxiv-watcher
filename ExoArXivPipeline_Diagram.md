exo-arxiv-watcher — agent architecture

# 3段階のエージェントパイプライン

Anthropic「Building Effective Agents」の3パターン(prompt chaining / evaluator-optimizer / routing)を、同じ天文学論文タグ付けタスクの上で段階的に積み上げた構成。

model: **Qwen2.5-7B-Instruct-Q8_0** backend: **llama-server (Vulkan, Arc 130V, ローカル)** framework: **any-agent (tinyagent)**

LLMエージェント呼び出し

データ(中間生成物・入出力)

Stage 1

## Prompt Chaining — stage1_chain.py

1つのタスクを直列の2ステップに分解。前段の出力(日本語要約)が、そのまま後段(タグ付け)の入力になる。

2回のLLM呼び出しが直列につながり、後段の入力は必ず前段の出力になる。

Stage 2

## Evaluator–Optimizer — stage2_evaluator_optimizer.py

Taggerの出力をEvaluatorが元のabstractと照合。不一致ならNGの理由を添えてTaggerに戻し、最大2回まで再試行する。

同じ7Bモデルに二値の「OK/NG」だけを問うと素通りしがちだったため、Evaluatorには候補タグを1件ずつ検討させてから結論を出させている。

Stage 3

## Routing — stage3_routing.py

本処理(Stage1+2)の前にRouterを置き、興味範囲外の論文はコストのかかるチェーンに進めずスキップする。

Stage1・Stage2のチェーンは「本処理」としてそのまま再利用し、Routerがその手前で通す/通さないだけを判定する。

個人開発プロジェクト exo-arxiv-watcher の手習い記録。Anthropic「Building Effective Agents」記事のパターンを、Mozilla.ai any-agent + ローカルLLM(llama.cpp / Qwen2.5-7B)で実装しながら確認している。