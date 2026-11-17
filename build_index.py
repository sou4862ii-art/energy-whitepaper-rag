"""
PDF から検索用の索引を作る。

使い方:
    python build_index.py                       # data/pdf/ のPDFすべて
    python build_index.py --pdf-dir data/pdf_test --index-dir data/index_test
    python build_index.py --embedder local      # ベクトル検索も使う（none / voyage / local）
"""

import argparse
import time
from pathlib import Path

from rag import config
from rag.embed import get_embedder
from rag.ingest import chunk_pdf
from rag.search import Index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(config.PDF_DIR))
    ap.add_argument("--index-dir", default=str(config.INDEX_DIR))
    ap.add_argument("--embedder", default=config.EMBEDDER, help="none / voyage / local")
    ap.add_argument("--chunk-chars", type=int, default=config.CHUNK_CHARS)
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"PDFがありません: {Path(args.pdf_dir).resolve()}")

    t0 = time.time()
    chunks = []
    for p in pdfs:
        cs = chunk_pdf(p, args.chunk_chars)
        print(f"  {p.name}: {len(cs)} チャンク")
        chunks.extend(cs)

    embedder = get_embedder(args.embedder)
    vectors = None
    if embedder is not None:
        print(f"埋め込み中（{embedder.name}）…")
        vectors = embedder.embed([f"{c.heading}\n{c.text}" for c in chunks], kind="document")

    Index(chunks, vectors).save(args.index_dir, embedder_name=args.embedder, pdf_dir=args.pdf_dir)
    print(f"完了: {len(pdfs)} ファイル, {len(chunks)} チャンク, "
          f"埋め込み={args.embedder}, {time.time() - t0:.1f} 秒 → {Path(args.index_dir).resolve()}")


if __name__ == "__main__":
    main()
