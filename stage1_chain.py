from any_agent import AgentConfig, AnyAgent

from tools.llm_config import get_model_config

MODEL_ID, API_BASE, API_KEY = get_model_config()

SUMMARIZE_INSTRUCTIONS = (
    "あなたは天文学論文の要約者です。"
    "与えられた英語のabstractを、日本語で3行以内に要約してください。"
    "専門用語は残してよいですが、平易な説明を心がけてください。"
    "要約以外の文章は出力しないでください。"
)

TAG_LIST = ["検出手法", "バイオシグネチャ", "大気科学", "軌道力学", "観測装置", "その他"]

TAG_INSTRUCTIONS = (
    "あなたは論文分類者です。与えられた要約を読み、次のタグ候補から"
    f"最も当てはまるものを1〜2個選んでください: {', '.join(TAG_LIST)}。"
    "出力はカンマ区切りのタグ名のみにし、他の文章は含めないでください。"
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


def main():
    from stage0_fetch import fetch_new_papers
    summarizer = build_agent(SUMMARIZE_INSTRUCTIONS)
    tagger = build_agent(TAG_INSTRUCTIONS)

    papers = fetch_new_papers(max_results=3)  # まずは3件だけで確認
    for p in papers:
        print(f"=== {p['title']} ===")

        summary_trace = summarizer.run(p["summary"])
        summary = summary_trace.final_output
        print(f"[要約] {summary}")

        tag_trace = tagger.run(summary)
        tags = tag_trace.final_output
        print(f"[タグ] {tags}")
        print()


if __name__ == "__main__":
    main()