from any_agent import AgentConfig, AnyAgent

from tools.llm_config import get_model_config

MODEL_ID, API_BASE, API_KEY = get_model_config()

TAG_LIST = ["検出手法", "バイオシグネチャ", "大気科学", "軌道力学", "観測装置", "その他"]
MAX_RETRIES = 2  # 初回 + 最大2回のやり直し

SUMMARIZE_INSTRUCTIONS = (
    "あなたは天文学論文の要約者です。"
    "与えられた英語のabstractを、日本語で3行以内に要約してください。"
    "専門用語は残してよいですが、平易な説明を心がけてください。"
    "要約以外の文章は出力しないでください。"
)

TAG_INSTRUCTIONS = (
    "あなたは論文分類者です。与えられた要約(と、あれば直前の査読コメント)を読み、次のタグ候補から"
    f"最も当てはまるものを1〜2個選んでください: {', '.join(TAG_LIST)}。"
    "出力はカンマ区切りのタグ名のみにし、他の文章は含めないでください。"
)

# EVALUATOR_INSTRUCTIONS = (
#     "あなたはタグ付けの査読者です。論文のabstract(英語)と、それに付けられたタグを見て妥当性を判定してください。"
#     f"タグ候補は次の通りです: {', '.join(TAG_LIST)}。"
#     "妥当なら「OK」とだけ出力してください。"
#     "妥当でない場合は「NG: <理由と、より適切だと思うタグ>」の形式で出力し、余計な文章は含めないでください。"
# )
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
)


def build_agent(instructions: str) -> AnyAgent:
    return AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id=MODEL_ID,
            api_key=API_KEY,
            api_base=API_BASE,
            instructions=instructions,
            tools=[],
        ),
    )


def tag_with_evaluation(abstract: str, jp_summary: str, tagger, evaluator):
    log = []
    feedback = ""
    tags = ""
    for attempt in range(1, MAX_RETRIES + 2):
        prompt = jp_summary if not feedback else f"{jp_summary}\n\n前回の査読コメント: {feedback}\nこれを踏まえて選び直してください。"
        tags = tagger.run(prompt).final_output.strip()

        eval_prompt = f"abstract: {abstract}\n\n付けられたタグ: {tags}"
        verdict_full = evaluator.run(eval_prompt).final_output.strip()
        verdict_last_line = verdict_full.strip().splitlines()[-1].strip()

        log.append({"attempt": attempt, "tags": tags, "verdict": verdict_full})
        if verdict_last_line.upper().startswith("OK"):
            break
        feedback = verdict_last_line

    return tags, log


def main():
    from stage0_fetch import fetch_new_papers

    summarizer = build_agent(SUMMARIZE_INSTRUCTIONS)
    tagger = build_agent(TAG_INSTRUCTIONS)
    evaluator = build_agent(EVALUATOR_INSTRUCTIONS)

    papers = fetch_new_papers(max_results=3)
    for p in papers:
        print(f"=== {p['title']} ===")
        jp_summary = summarizer.run(p["summary"]).final_output
        print(f"[要約] {jp_summary}")

        final_tags, log = tag_with_evaluation(p["summary"], jp_summary, tagger, evaluator)
        for entry in log:
            print(f"  試行{entry['attempt']}: タグ={entry['tags']} / 査読={entry['verdict']}")
        print(f"[最終タグ] {final_tags}\n")


if __name__ == "__main__":
    main()