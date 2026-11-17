"""
RAG の精度を数字で測る。

質問セット（CSV）の列:
    id, question, expected_pages, answerable, memo
      expected_pages : 答えが書かれているPDFのページ番号。複数なら ; 区切り（例 "45;46"）
      answerable     : yes = 白書に答えがある / no = 答えがない（「見つからない」と答えるべき質問）

使い方:
    python eval/evaluate.py eval/questions.csv                 # 検索の精度だけ（APIキー不要）
    python eval/evaluate.py eval/questions.csv --with-answer   # Claude の回答・引用まで評価

指標:
    検索  hit@1 / hit@k : 正解ページの抜粋が 1位 / 上位k件 に入った割合
          MRR           : 正解が何位に出たかの逆数の平均（1.0 が最高）
    回答  回答率         : 答えがある質問に、引用付きで答えられた割合
          引用の正確さ   : 付いた引用のうち、正解ページを指していた割合
          正しい「不明」 : 答えがない質問に、引用なしで「見つからない」と答えた割合
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import config  # noqa: E402
from rag.answer import ask  # noqa: E402
from rag.embed import get_embedder  # noqa: E402
from rag.search import Index  # noqa: E402


def load_questions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["expected"] = {int(p) for p in r.get("expected_pages", "").replace(" ", "").split(";") if p}
        r["answerable"] = r.get("answerable", "yes").strip().lower() != "no"
    return rows


def evaluate(rows, index, top_k=8, client=None, with_answer=False):
    results = []
    for r in rows:
        res = {"id": r["id"], "question": r["question"], "answerable": r["answerable"]}
        retrieved = index.search(r["question"], top_k=top_k)
        pages = [c.page for c, _ in retrieved]
        res["retrieved_pages"] = pages
        if r["answerable"] and r["expected"]:
            rank = next((i + 1 for i, p in enumerate(pages) if p in r["expected"]), None)
            res.update(hit1=rank == 1, hitk=rank is not None, rr=1 / rank if rank else 0.0)
        if with_answer:
            ans = ask(r["question"], index, client=client, top_k=top_k)
            cited = [c.page for c in ans.citations]
            res.update(answer=ans.text, cited_pages=cited, found=ans.found)
            if r["answerable"]:
                res["answered"] = ans.found and bool(cited)
                if cited and r["expected"]:
                    res["citation_precision"] = sum(p in r["expected"] for p in cited) / len(cited)
            else:
                res["correct_refusal"] = (not ans.found) and not cited
        results.append(res)
    return results


def summarize(results, top_k):
    def mean(key):
        vals = [float(r[key]) for r in results if key in r]
        return (round(sum(vals) / len(vals), 3), len(vals)) if vals else (None, 0)

    s = {}
    for key, label in [("hit1", "検索 hit@1"), ("hitk", f"検索 hit@{top_k}"), ("rr", "検索 MRR"),
                       ("answered", "回答率"), ("citation_precision", "引用の正確さ"),
                       ("correct_refusal", "正しい「不明」")]:
        value, n = mean(key)
        if n:
            s[label] = {"値": value, "対象の質問数": n}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--index-dir", default=str(config.INDEX_DIR))
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--with-answer", action="store_true", help="Claudeを呼んで回答と引用も評価")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    meta = Index.read_meta(args.index_dir)
    index = Index.load(args.index_dir, get_embedder(meta.get("embedder", "none")))
    rows = load_questions(args.questions)
    results = evaluate(rows, index, args.top_k, with_answer=args.with_answer)
    summary = summarize(results, args.top_k)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.questions).stem
    (out / f"{name}_summary.json").write_text(json.dumps(
        {"索引": meta, "top_k": args.top_k, "結果": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / f"{name}_detail.csv", "w", encoding="utf-8-sig", newline="") as f:
        keys = sorted({k for r in results for k in r}, key=lambda k: list(results[0]).index(k)
                      if k in results[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    print(f"質問 {len(rows)} 件（埋め込み: {meta.get('embedder')}, top_k={args.top_k}）")
    for label, v in summary.items():
        print(f"  {label:<12} {v['値']:.3f}  （{v['対象の質問数']} 問）")
    print(f"詳細: {out / f'{name}_detail.csv'}")


if __name__ == "__main__":
    main()
