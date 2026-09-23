import subprocess
import xml.etree.ElementTree as ET

ATOM_NS = "{http://www.w3.org/2005/Atom}"

USER_AGENT = "exo-arxiv-watcher/0.1 (personal script; contact: taka@darkhaloes.com)"


def fetch_new_papers(category="astro-ph.EP", max_results=20):
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=cat:{category}"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    result = subprocess.run(
        ["curl", "-s", "-H", f"User-Agent: {USER_AGENT}", url],
        capture_output=True,
        text=True,
        check=True,
    )
    root = ET.fromstring(result.stdout)

    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        papers.append({
            "id": entry.find(f"{ATOM_NS}id").text.strip(),
            "title": entry.find(f"{ATOM_NS}title").text.strip().replace("\n", " "),
            "summary": entry.find(f"{ATOM_NS}summary").text.strip().replace("\n", " "),
            "published": entry.find(f"{ATOM_NS}published").text.strip(),
        })
    return papers


def main():
    papers = fetch_new_papers()
    print(f"取得件数: {len(papers)}\n")
    for p in papers:
        print(f"- [{p['published'][:10]}] {p['title']}")
        print(f"  {p['id']}")


if __name__ == "__main__":
    main()