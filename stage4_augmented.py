import os

from any_agent import AgentCancel, AgentConfig, AnyAgent
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


class MaxToolCallsExceeded(AgentCancel):
    """1回のrun()内でツール呼び出しが上限を超えた場合に送出される。

    ツール未装備だったStage3までは起こり得なかった「モデルがツール呼び出しを
    延々と繰り返して終わらない」暴走を防ぐガード。
    """


class ToolCallLogger(Callback):
    """ツール呼び出しをデモ資料用に標準出力へ記録し、暴走を防ぐ薄いラッパー。"""

    def __init__(self, max_tool_calls: int = 5):
        self.max_tool_calls = max_tool_calls

    def before_tool_execution(self, context, *args, **kwargs):
        count = context.shared.get("tool_call_count", 0) + 1
        context.shared["tool_call_count"] = count
        if count > self.max_tool_calls:
            raise MaxToolCallsExceeded(f"tool called {count} times (limit {self.max_tool_calls})")
        return context

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
        # abstract(英語)も渡すのは、find_similar_casesが履歴(英語abstract)と
        # 比較可能な言語のテキストをTaggerから受け取れるようにするため。
        base_prompt = f"abstract: {abstract}\n\n{jp_summary}"
        prompt = base_prompt if not feedback else f"{base_prompt}\n\n前回の査読コメント: {feedback}\nこれを踏まえて選び直してください。"
        try:
            tags = tagger.run(prompt).final_output.strip()
            eval_prompt = f"paper_id (URL): {paper_id}\nabstract: {abstract}\n\n付けられたタグ: {tags}"
            verdict_full = evaluator.run(eval_prompt).final_output.strip()
        except MaxToolCallsExceeded as e:
            verdict_full = f"NG: ツール呼び出し上限に達したため打ち切り ({e})"

        verdict_last_line = verdict_full.strip().splitlines()[-1].strip()
        log.append({"attempt": attempt, "tags": tags, "verdict": verdict_full})

        if verdict_last_line.upper().startswith("OK"):
            record_case(paper_id, title, abstract, tags)
            break
        if "ツール呼び出し上限" in verdict_full:
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
