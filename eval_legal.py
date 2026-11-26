"""Before/after evaluation on YOUR legal RAG assistant.

Step A - collect: run each question through the guard API (which calls your RAG)
    python scripts/eval_legal.py collect --questions data/legal_questions.txt --out legal_eval.csv

Step B - label by hand in Excel/Sheets: fill two columns with 1 (hallucinated) or 0 (faithful)
    orig_hallucinated   -> was the RAG's ORIGINAL answer hallucinated?
    final_hallucinated  -> is what the USER finally sees hallucinated? (blocked answers = 0)

Step C - score:
    python scripts/eval_legal.py score --csv legal_eval.csv
"""
import argparse
import json

import httpx
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


def collect(questions_path, api, out):
    with open(questions_path, encoding="utf-8") as f:
        questions = [q.strip() for q in f if q.strip()]
    rows = []
    for q in tqdm(questions):
        r = httpx.post(f"{api}/ask", json={"question": q}, timeout=300).json()
        rows.append({
            "question": q,
            "original_answer": r["original_answer"],
            "final_answer": r["final_answer"],
            "decision": r["decision"],
            "faithfulness": r["faithfulness"],
            "n_unsupported": r["n_unsupported"],
            "guard_latency_ms": r["guard_latency_ms"],
            "unsupported_claims": json.dumps([c["text"] for c in r["claims"]
                                              if c["label"] != "supported"], ensure_ascii=False),
            "contexts": json.dumps(r["contexts"], ensure_ascii=False),
            "orig_hallucinated": "",
            "final_hallucinated": "",
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} rows to {out}. Now fill the two *_hallucinated columns by hand.")


def score(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["orig_hallucinated", "final_hallucinated"])
    before = df["orig_hallucinated"].mean()
    after = df["final_hallucinated"].mean()
    flagged = df["decision"] != "pass"   # guard said "something is wrong"
    p, r, f1, _ = precision_recall_fscore_support(df["orig_hallucinated"].astype(int), flagged,
                                                  average="binary", zero_division=0)
    print(f"Answers labelled:            {len(df)}")
    print(f"Hallucination rate BEFORE:   {before:.1%}")
    print(f"Hallucination rate AFTER:    {after:.1%}")
    print(f"Detector precision/recall/F1: {p:.2f} / {r:.2f} / {f1:.2f}")
    print(f"Blocked answers:             {(df['decision'] == 'block').mean():.1%}")
    print(f"Mean guard latency:          {df['guard_latency_ms'].mean():.0f} ms")
    print("\nDecision counts:\n", df["decision"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--questions", required=True)
    c.add_argument("--api", default="http://localhost:8000")
    c.add_argument("--out", default="legal_eval.csv")
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True)
    args = ap.parse_args()
    collect(args.questions, args.api, args.out) if args.cmd == "collect" else score(args.csv)
