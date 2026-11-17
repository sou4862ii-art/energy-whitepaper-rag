"""
PDF を「ページ番号と見出しの位置を保ったチャンク」に分ける。

白書のPDFならではの処理:
  - 2段組みを判定し、左の段 → 右の段の順に読む（段をまたぐ図表や見出しで区切りながら）
  - ページ上端の柱（部・章の名前）、下端のページ番号、余白の縦書きタブを位置で除去
  - 全角数字などを NFKC で正規化（「第２章」→「第2章」）し、文字化けの記号を除く
  - 2行に分かれた見出し（「第1 節」＋題名）を1つにまとめ、「第○部 / 章 / 節」の階層を追跡
  - 目次ページは見出しの形の行が並ぶので、索引から除く
  - PDFの行末で途切れた日本語の文をつなぎ直し、字下げで段落を区切る
  - チャンクはページをまたがない（引用元のページを正確に示すため）
"""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf

HEADING_LEVELS = [
    ("part", re.compile(r"^第\s*(\d+)\s*部\s*(.*)$")),
    ("chapter", re.compile(r"^第\s*(\d+)\s*章\s*(.*)$")),
    ("section", re.compile(r"^第\s*(\d+)\s*節\s*(.*)$")),
]
SENTENCE_END = ("。", "」", "）", ")", "：", ":")
JUNK_CHARS = re.compile(r"[ᰀ-ᱏ-]")   # フォント由来の文字化け・私用領域

# ページ上の位置（ページの幅・高さに対する割合）
HEADER_ZONE, FOOTER_ZONE = 0.065, 0.95
MARGIN_ZONE = 0.07


@dataclass
class Chunk:
    id: str
    doc: str          # PDFファイル名
    page: int         # PDFのページ番号（1始まり。ビューアの #page=N と一致）
    heading: str      # 例: "第2部 ... > 第3章 地域と共生した再生可能エネルギーの導入拡大 > 第1節 ..."
    text: str


def normalize(s: str) -> str:
    s = JUNK_CHARS.sub("", unicodedata.normalize("NFKC", s))
    return re.sub(r"[ \t　]+", " ", s).strip()


def _heading(text):
    """ブロックの文字列が見出しなら (階層, 見出し文字列) を返す。"""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) > 70 or "。" in t:
        return None
    for level, pat in HEADING_LEVELS:
        m = pat.match(t)
        if m:
            title = m.group(2).strip()
            if re.match(r"^(では|は|の|に|で|を|が|と|へ|も|において|及び|まで)", title):
                return None          # 「第2部では、…」のような本文の文
            return level, f"第{m.group(1)}{ {'part': '部', 'chapter': '章', 'section': '節'}[level] } {title}".strip()
    return None


SUBHEADING = re.compile(r"^(\d{1,2}[.．]|\(\d{1,2}\)|[①-⑳])\s*\S")


def _subheading(lines):
    """「4.原子力災害の…」「(2)エネルギー価格への影響」のような番号付きの小見出し。"""
    t = "".join(t for _, t in lines)
    if len(lines) <= 3 and len(t) <= 60 and SUBHEADING.match(t) and not t.endswith("。"):
        return t
    return None


def _blocks(page):
    """本文のテキストブロックを [(x0, y0, x1, y1, [行...])] で返す（柱・ページ番号・余白タブは除く）。"""
    W, H = page.rect.width, page.rect.height
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        if y1 <= H * HEADER_ZONE or y0 >= H * FOOTER_ZONE:
            continue
        if x1 <= W * MARGIN_ZONE or x0 >= W * (1 - MARGIN_ZONE):
            continue
        lines = []
        for line in b["lines"]:
            t = normalize("".join(s["text"] for s in line["spans"]))
            if t:
                lines.append((line["bbox"][0], t))
        if lines:
            out.append((x0, y0, x1, y1, lines))
    return out


def _reading_order(blocks, width):
    """2段組みの読み順に並べる。段をまたぐブロック（大見出し・横長の図表）で上下に区切り、
    各区間の中では 左の段 → 右の段 の順にする。"""
    mid = width / 2
    def column(b):
        x0, _, x1, _, _ = b
        if x1 <= mid + 15:
            return "left"
        if x0 >= mid - 15:
            return "right"
        return "full"

    fulls = sorted((b for b in blocks if column(b) == "full"), key=lambda b: b[1])
    others = [b for b in blocks if column(b) != "full"]
    ordered, top = [], -1.0
    for f in fulls + [None]:
        bottom = f[1] if f else float("inf")
        band = [b for b in others if top <= b[1] < bottom]
        for col in ("left", "right"):
            ordered += sorted((b for b in band if column(b) == col), key=lambda b: b[1])
        if f:
            ordered.append(f)
            top = f[1]
    return ordered


def extract_pages(pdf_path):
    """[(ページ番号, [項目...])]。項目は ("heading", 階層, 見出し) か ("line", 文字列, 字下げか)。
    目次ページ（見出しの形の行が多いページ）は空にする。"""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        items = []
        blocks = _reading_order(_blocks(page), page.rect.width)
        n_heads = 0
        for x0, _, x1, _, lines in blocks:
            h = _heading("".join(t for _, t in lines)) if len(lines) <= 3 else None
            if h:
                items.append(("heading", *h))
                n_heads += 1
                continue
            sub = _subheading(lines)
            if sub:
                items.append(("line", sub, "sub"))
                continue
            left = min(x for x, _ in lines)
            for lx, t in lines:
                items.append(("line", t, lx - left >= 6))
        pages.append((i + 1, [] if n_heads >= 5 else items))

    # 位置で取り切れなかった繰り返し行（全ページ共通の注記など）も除く
    n = len(pages)
    counts = Counter(t for _, items in pages for t in {it[1] for it in items if it[0] == "line"})
    repeated = {t for t, c in counts.items() if n >= 4 and c / n >= 0.3}
    return [(no, [it for it in items if not (it[0] == "line" and it[1] in repeated)]) for no, items in pages]


def _paragraphs(lines):
    """[(文字列, 字下げか)] を段落にまとめる。字下げで始まる行と、句点で終わる行で区切る。"""
    paras, buf = [], ""
    for text, indented in lines:
        if indented == "sub":                  # 小見出しは独立した段落にする
            if buf:
                paras.append(buf)
                buf = ""
            paras.append(text)
            continue
        if indented and buf:
            paras.append(buf)
            buf = ""
        if buf and re.match(r"[A-Za-z0-9]", text[0]) and re.search(r"[A-Za-z0-9]$", buf):
            buf += " "
        buf += text
        if text.endswith(SENTENCE_END) and not indented and len(text) < 20:
            # 句点で終わる短い行 = 段落の最終行
            paras.append(buf)
            buf = ""
    if buf:
        paras.append(buf)
    return paras


def chunk_pdf(pdf_path, max_chars=500):
    pdf_path = Path(pdf_path)
    path = {"part": "", "chapter": "", "section": ""}
    chunks = []

    def heading_str():
        return " > ".join(v for v in path.values() if v)

    for page_no, items in extract_pages(pdf_path):
        n = 0
        buf, heading = [], heading_str()

        def flush(lines):
            nonlocal n, buf
            for para in _paragraphs(lines):
                if buf and sum(len(p) for p in buf) + len(para) > max_chars:
                    emit()
                buf.append(para)

        def emit():
            nonlocal n, buf
            text = "\n".join(buf).strip()
            if len(text) >= 20:
                chunks.append(Chunk(f"{pdf_path.stem}-p{page_no}-{n}", pdf_path.name, page_no, heading, text))
                n += 1
            buf = []

        pending = []
        for it in items:
            if it[0] == "heading":
                flush(pending)
                pending = []
                emit()
                _, level, title = it
                path[level] = title
                if level == "part":
                    path["chapter"] = path["section"] = ""
                elif level == "chapter":
                    path["section"] = ""
                heading = heading_str()
            else:
                pending.append((it[1], it[2]))
        flush(pending)
        emit()
    return chunks


def save_chunks(chunks, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
