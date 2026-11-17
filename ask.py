"""
白書に質問する（コマンドライン版）。

使い方:
    python ask.py "地熱発電の導入が進まない理由は？"
    python ask.py "..." --dry-run      # APIを呼ばず、検索結果だけ表示（キー不要）
    python ask.py                      # 対話モード（空行で終了）
"""

import argparse
import sys

from rag import config
from rag.answer import ask, format_answer
from rag.embed import get_embedder
from rag.search import Index


def show_retrieved(retrieved):
    for rank, (c, info) in enumerate(retrieved, 1):
        print(f"{rank:2d}. {c.doc} {c.page}ページ  {c.heading}")
        print(f"    キーワード順位={info['bm25_rank'] or '－'}  ベクトル順位={info['vector_rank'] or '－'}")
        print(f"    {c.text[:90].replace(chr(10), ' ')}…")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--index-dir", default=str(config.INDEX_DIR))
    ap.add_argument("--embedder", default=None, help="索引作成時と同じもの（none / voyage / local）")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--dry-run", action="store_true", help="Claudeを呼ばずに検索結果だけ表示")
    args = ap.parse_args()

    # 埋め込みは、指定がなければ索引を作ったときと同じものを使う
    embedder = args.embedder or Index.read_meta(args.index_dir).get("embedder", "none")
    index = Index.load(args.index_dir, get_embedder(embedder))
    questions = [args.question] if args.question else iter(lambda: input("\n質問> ").strip(), "")
    for q in questions:
        if args.dry_run:
            show_retrieved(index.search(q, top_k=args.top_k))
            continue
        try:
            ans = ask(q, index, top_k=args.top_k)
        except Exception as e:  # キー未設定・通信エラーなどを分かりやすく
            sys.exit(f"Claude API の呼び出しに失敗しました: {e}\n"
                     "ANTHROPIC_API_KEY が .env か環境変数に設定されているか確認してください。")
        print("\n" + format_answer(ans))
        if ans.usage:
            print(f"\n（トークン: 入力 {ans.usage['input_tokens']} / 出力 {ans.usage['output_tokens']}）")


if __name__ == "__main__":
    main()
