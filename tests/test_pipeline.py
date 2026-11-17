"""
APIキーなしで通しの動作を確かめるテスト。

Claude の応答は、公式ドキュメントの search_result_location 形式を真似た偽物を使う。
    python -m pytest tests -q      （pytest が無ければ  python tests/test_pipeline.py）
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.answer import NOT_FOUND, ask, build_request, format_answer  # noqa: E402
from rag.ingest import chunk_pdf  # noqa: E402
from rag.search import Index, rrf  # noqa: E402

PDF = ROOT / "data" / "pdf_test" / "test_whitepaper.pdf"


def _index():
    if not PDF.exists():
        from tests.make_test_pdf import main as make
        make(PDF)
    return Index(chunk_pdf(PDF))


class FakeClient:
    """messages.create を受け取り、渡された1番目の検索結果の3文目を引用して答える。"""

    def __init__(self, not_found=False):
        self.not_found = not_found
        self.last_request = None
        self.messages = self

    def create(self, **req):
        self.last_request = req
        if self.not_found:
            return NS(content=[NS(type="text", text=NOT_FOUND, citations=None)],
                      usage=NS(input_tokens=100, output_tokens=20))
        sr = req["messages"][0]["content"][0]
        cite = NS(type="search_result_location", source=sr["source"], title=sr["title"],
                  cited_text=sr["content"][2]["text"], search_result_index=0,
                  start_block_index=2, end_block_index=3)
        return NS(content=[NS(type="text", text="主な理由は三つあります。", citations=None),
                           NS(type="text", text="掘削の成功率が低く費用がかさむことなどです。", citations=[cite])],
                  usage=NS(input_tokens=1200, output_tokens=80))


def test_chunks_have_pages_and_headings():
    chunks = chunk_pdf(PDF)
    assert len(chunks) == 4
    geo = next(c for c in chunks if "地熱" in c.text)
    assert geo.page == 3
    assert geo.heading.endswith("第1節 地熱発電")
    assert "テスト用ダミー文書" not in geo.text            # ヘッダーが除去されている
    assert "世界有数とされるが、導入量" in geo.text          # 行またぎがつながっている


def test_search_finds_right_page():
    ix = _index()
    assert ix.search("地熱発電が増えない理由", top_k=1)[0][0].page == 3
    assert ix.search("出力制御", top_k=1)[0][0].page == 4


def test_rrf_merges_rankings():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0] == 1


def test_vector_path_with_fake_embedder():
    class Fake:
        name = "fake"

        def embed(self, texts, kind="document"):
            v = np.array([[("地熱" in t) * 1.0, ("太陽" in t) * 1.0, 0.1] for t in texts])
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    chunks = chunk_pdf(PDF)
    emb = Fake()
    ix = Index(chunks, emb.embed([c.heading + c.text for c in chunks]), emb)
    top = ix.search("地熱", top_k=1)[0]
    assert top[0].page == 3 and top[1]["vector_rank"] == 1


def test_request_uses_search_result_blocks():
    ix = _index()
    req = build_request("地熱発電の課題は？", ix.search("地熱発電の課題", top_k=2))
    first = req["messages"][0]["content"][0]
    assert first["type"] == "search_result"
    assert first["source"].endswith("#page=3")
    assert first["citations"] == {"enabled": True}
    assert len(first["content"]) >= 2                       # 文ごとのブロックに分割
    assert req["messages"][0]["content"][-1]["type"] == "text"


def test_answer_with_citation():
    ix = _index()
    ans = ask("地熱発電が増えない理由は？", ix, client=FakeClient())
    assert ans.found and ans.text.endswith("[1]")
    assert ans.citations[0].page == 3
    assert "導入が進まない主な理由" in ans.citations[0].cited_text
    assert "出典:" in format_answer(ans)


def test_not_found():
    ans = ask("月の裏側の天気は？", _index(), client=FakeClient(not_found=True))
    assert not ans.found and not ans.citations


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} 件すべて成功")
