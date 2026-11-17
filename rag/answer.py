"""
検索結果を Claude に渡し、引用付きで答えさせる。

Claude API の「search_result」コンテンツブロックを使う。検索で見つかった各チャンクを
search_result として渡すと、Claude は回答の各部分に「どの資料のどこを根拠にしたか」を
citations として返す。引用は API が保証する仕組みなので、根拠のない引用は付かない。

各チャンクは文ごとに小さなテキストブロックへ分けて渡す（引用の最小単位がブロックのため、
こうするとページ全体ではなく「その文」を引用として示せる）。
"""

import re
from dataclasses import dataclass, field

from . import config

NOT_FOUND = "白書の中に、この質問に答える記述は見つかりませんでした。"

SYSTEM_PROMPT = f"""あなたはエネルギー白書の内容について答えるアシスタントです。

規則:
- 与えられた検索結果（白書の抜粋）に書かれている内容だけを根拠に、日本語で答えてください。
- 数値・年度・固有名詞は、抜粋に書かれているとおりに正確に書いてください。計算や推測で数値を作らないでください。
- 抜粋に答えがない場合は、推測で補わず「{NOT_FOUND}」とだけ答えてください。
- 一部だけ分かる場合は、分かる範囲を答えたうえで、白書に記述がない点をはっきり書いてください。
- まず結論を1〜2文で述べ、必要なら箇条書きで補足してください。全体で400字程度までにしてください。
"""


@dataclass
class Citation:
    number: int
    doc: str
    page: int
    heading: str
    cited_text: str


@dataclass
class Answer:
    text: str                      # 引用番号 [1] などを埋め込んだ回答本文（Markdown）
    citations: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)   # [(Chunk, 情報)]
    usage: dict = field(default_factory=dict)
    found: bool = True


def _sentences(text):
    parts = re.split(r"(?<=[。！？])", text.replace("\n", ""))
    return [p for p in (s.strip() for s in parts) if p]


def to_search_results(retrieved):
    blocks = []
    for chunk, _ in retrieved:
        blocks.append({
            "type": "search_result",
            "source": f"{chunk.doc}#page={chunk.page}",
            "title": f"{chunk.heading or chunk.doc}（{chunk.page}ページ）",
            "content": [{"type": "text", "text": s} for s in _sentences(chunk.text)] or
                       [{"type": "text", "text": chunk.text}],
            "citations": {"enabled": True},
        })
    return blocks


def build_request(question, retrieved, model=None):
    return dict(
        model=model or config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            *to_search_results(retrieved),
            {"type": "text", "text": f"質問: {question}"},
        ]}],
    )


def parse_response(response, retrieved):
    """Claude の応答から、本文と引用（番号付き脚注）を組み立てる。"""
    parts, citations, seen = [], [], {}
    for block in response.content:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        marks = []
        for c in getattr(block, "citations", None) or []:
            if getattr(c, "type", None) != "search_result_location":
                continue
            key = (c.search_result_index, c.start_block_index, c.end_block_index)
            if key not in seen:
                chunk = retrieved[c.search_result_index][0]
                seen[key] = len(citations) + 1
                citations.append(Citation(seen[key], chunk.doc, chunk.page, chunk.heading,
                                          c.cited_text.strip()))
            marks.append(seen[key])
        if marks:
            parts.append("".join(f"[{n}]" for n in sorted(set(marks))))
    text = "".join(parts).strip()
    usage = getattr(response, "usage", None)
    return Answer(
        text=text,
        citations=citations,
        retrieved=retrieved,
        usage={"input_tokens": getattr(usage, "input_tokens", None),
               "output_tokens": getattr(usage, "output_tokens", None)} if usage else {},
        found=NOT_FOUND not in text,
    )


def ask(question, index, client=None, top_k=None, model=None):
    """質問 → 検索 → Claude → 引用付きの Answer。"""
    retrieved = index.search(question, top_k=top_k or config.TOP_K)
    if not retrieved:
        return Answer(text=NOT_FOUND, retrieved=[], found=False)
    if client is None:
        import anthropic
        client = anthropic.Anthropic()          # 環境変数 ANTHROPIC_API_KEY を使う
    response = client.messages.create(**build_request(question, retrieved, model))
    return parse_response(response, retrieved)


def format_answer(ans):
    """端末表示・Markdown用の整形。"""
    lines = [ans.text, ""]
    if ans.citations:
        lines.append("出典:")
        for c in ans.citations:
            snippet = c.cited_text if len(c.cited_text) <= 80 else c.cited_text[:80] + "…"
            lines.append(f"  [{c.number}] {c.doc} {c.page}ページ（{c.heading}）")
            lines.append(f"      「{snippet}」")
    return "\n".join(lines)
