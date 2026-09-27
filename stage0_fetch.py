import os
import platform
import shutil
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

ATOM_NS = "{http://www.w3.org/2005/Atom}"

USER_AGENT = "exo-arxiv-watcher/0.1 (personal script; contact: taka@darkhaloes.com)"

# export.arxiv.org appears to reject fresh (non-cached) requests from WSL2's
# OpenSSL-based curl (TLS fingerprint), while Windows' native Schannel-based
# curl.exe from the same host/IP succeeds. Prefer curl.exe under WSL2.
_WINDOWS_CURL = "/mnt/c/Windows/System32/curl.exe"


def _curl_path() -> str:
    if "microsoft" in platform.uname().release.lower() and os.path.exists(_WINDOWS_CURL):
        return _WINDOWS_CURL
    return shutil.which("curl") or "curl"


def fetch_new_papers(category="astro-ph.EP", max_results=20):
    query = urlencode({
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    })
    url = f"https://export.arxiv.org/api/query?{query}"
    result = subprocess.run(
        [_curl_path(), "-s", "-H", f"User-Agent: {USER_AGENT}", url],
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