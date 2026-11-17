"""引用したページを画像にし、引用された文に色を付ける（Web画面用）。"""

import re
import unicodedata

import pymupdf


def _norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def _page_chars(page):
    """ページ上の文字を [(正規化した文字, 文字の矩形)] で返す（空白は除く）。
    PDFの文字は全角・半角や改行位置が元の文章と違うため、文字単位で正規化して照合する。"""
    out = []
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    for c in _norm(ch["c"]):
                        out.append((c, pymupdf.Rect(ch["bbox"])))
    return out


def _line_rects(rects):
    """文字の矩形を行ごとにまとめる（色付けを1行1つの帯にする）。"""
    lines = []
    for r in rects:
        if lines and abs(lines[-1].y0 - r.y0) < r.height * 0.5 and r.x0 >= lines[-1].x0:
            lines[-1] |= r
        else:
            lines.append(pymupdf.Rect(r))
    return lines


def find_rects(page, text):
    chars = _page_chars(page)
    hay = "".join(c for c, _ in chars)
    target = _norm(text)
    # 引用文がそのまま見つからなければ、先頭から少しずつ短くして探す
    for length in (len(target), 60, 30, 15):
        probe = target[:length]
        if len(probe) < 8:
            break
        pos = hay.find(probe)
        if pos >= 0:
            return _line_rects([r for _, r in chars[pos:pos + len(probe)]])
    return []


def render_page(pdf_path, page_no, cited_texts=(), zoom=1.6):
    doc = pymupdf.open(pdf_path)
    page = doc[page_no - 1]
    for cited in cited_texts:
        for rect in find_rects(page, cited):
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(1.0, 0.85, 0.2))
            annot.update()
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return pix.tobytes("png")
