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
