"""
ハイブリッド検索: キーワード検索（BM25）＋ ベクトル検索（任意）を RRF で統合する。

日本語は単語の区切りがないため、キーワード検索は「文字2文字ずつ（bigram）」で索引を作る。
形態素解析の辞書が要らず、「GX」「ALPS処理水」「第213-1-1」のような固有名詞・番号も取りこぼしにくい。
ベクトル検索は「言い換え」に強い（例: 「地熱が増えない理由」⇔「導入が進まない要因」）。
両方の順位を Reciprocal Rank Fusion（RRF）で合わせると、互いの弱点を補える。
"""

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ingest import Chunk, load_chunks


def _tokens(text):
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(_tokens(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean() if len(self.len) else 0.0
        df = Counter(tok for t in self.tf for tok in t)
        n = len(docs)
        self.idf = {tok: math.log(1 + (n - c + 0.5) / (c + 0.5)) for tok, c in df.items()}
        self.postings = defaultdict(list)
        for i, t in enumerate(self.tf):
            for tok, c in t.items():
                self.postings[tok].append((i, c))

    def scores(self, query):
        s = np.zeros(len(self.tf))
        for tok in set(_tokens(query)):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, c in self.postings[tok]:
                denom = c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s[i] += idf * c * (self.k1 + 1) / denom
        return s


def rrf(rankings, k=60):
    """複数の順位リスト（チャンク番号の並び）を1つに統合する。"""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] += 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


class Index:
    """チャンク・BM25・埋め込みをまとめて保存/読み込みする。"""

    def __init__(self, chunks, vectors=None, embedder=None):
        self.chunks = chunks
        self.bm25 = BM25([f"{c.heading}\n{c.text}" for c in chunks])
        self.vectors = vectors
        self.embedder = embedder

    # ---- 保存と読み込み
    def save(self, index_dir, embedder_name="none", pdf_dir=None):
        from .ingest import save_chunks
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, d / "chunks.jsonl")
        if self.vectors is not None:
            np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps(
            {"embedder": embedder_name, "n_chunks": len(self.chunks),
             "pdf_dir": str(Path(pdf_dir).resolve()) if pdf_dir else None,
             "docs": sorted({c.doc for c in self.chunks})}, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def read_meta(index_dir):
        p = Path(index_dir) / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    @classmethod
    def load(cls, index_dir, embedder=None):
        d = Path(index_dir)
        if not (d / "chunks.jsonl").exists():
            raise FileNotFoundError(f"索引がありません: {d}（先に build_index.py を実行してください）")
        chunks = load_chunks(d / "chunks.jsonl")
        vec_path = d / "vectors.npy"
        vectors = np.load(vec_path) if vec_path.exists() and embedder is not None else None
        return cls(chunks, vectors, embedder)

    # ---- 検索
    def search(self, query, top_k=8, pool=40):
        """[(Chunk, 情報dict)] を返す。情報には各方式での順位を入れる（評価・説明用）。"""
        bm = self.bm25.scores(query)
        bm_rank = [i for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        rankings = [bm_rank]
        vec_rank = []
        if self.vectors is not None and self.embedder is not None:
            q = self.embedder.embed([query], kind="query")[0]
            sim = self.vectors @ q
            vec_rank = list(np.argsort(-sim)[:pool])
            rankings.append(vec_rank)
        merged = rrf(rankings)[:top_k]
        out = []
        for i in merged:
            out.append((self.chunks[i], {
                "bm25_rank": bm_rank.index(i) + 1 if i in bm_rank else None,
                "vector_rank": vec_rank.index(i) + 1 if i in vec_rank else None,
            }))
        return out
