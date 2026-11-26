"""Benchmark the three detectors on RAGTruth.

Get the data first:
    git clone https://github.com/ParticleMedia/RAGTruth
Then:
    python scripts/eval_ragtruth.py --data_dir RAGTruth/dataset --methods nli,uq --n_test 300
    python scripts/eval_ragtruth.py --data_dir RAGTruth/dataset --methods selfcheck --n_test 100

What it reports (response level: "does this answer contain any hallucination?"
and claim level: "is this sentence part of a hallucinated span?"):
    AUROC (threshold-free), and Precision / Recall / F1 at a threshold tuned on a
    dev sample from the TRAIN split (never tune on the test set).
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_guard.claims import split_claims  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_context(src: dict) -> str:
    info = src.get("source_info")
    if isinstance(info, str):
        return info                                   # Summary task
    if isinstance(info, dict):
        if "passages" in info:
            return info["passages"]                   # QA task
        return json.dumps(info, ensure_ascii=False)   # Data2txt task (JSON records)
    return src.get("prompt", "")


def overlaps(claim, spans):
    return any(claim.start < s["end"] and claim.end > s["start"] for s in spans)


def best_threshold(scores, labels):
    best_t, best_f1 = 0.5, -1
    for t in np.unique(np.round(scores, 3)):
        p, r, f1, _ = precision_recall_fscore_support(labels, scores >= t, average="binary",
                                                      zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def build_detectors(methods):
    from rag_guard.nli import NLIScorer
    dets = {}
    nli = NLIScorer()
    if "nli" in methods:
        dets["nli"] = lambda ex, claims: [1 - s["entailment"] for s in
                                           nli.score_claims([c.text for c in claims], [ex["context"]])]
    if "selfcheck" in methods:
        from rag_guard.llm import LLM
        from rag_guard.self_consistency import SelfConsistencyChecker
        sc = SelfConsistencyChecker(LLM(), nli, n_samples=5)
        dets["selfcheck"] = lambda ex, claims: [s["hallucination_score"] for s in
                                                sc.score_claims([c.text for c in claims], ex["prompt"])]
    if "uq" in methods:
        from rag_guard.uncertainty import TokenUncertainty
        uq = TokenUncertainty()
        dets["uq"] = lambda ex, claims: [s["hallucination_score"] for s in
                                         uq.score_claims(claims, ex["prompt"], ex["response"])]
    return dets


def run(examples, dets):
    """Returns rows: one per claim, with every method's score and gold label."""
    rows = []
    for ex in tqdm(examples):
        claims = split_claims(ex["response"])
        if not claims:
            continue
        per_method, times = {}, {}
        for name, fn in dets.items():
            t0 = time.perf_counter()
            per_method[name] = fn(ex, claims)
            times[name] = (time.perf_counter() - t0) * 1000
        for k, c in enumerate(claims):
            row = {"id": ex["id"], "task": ex["task"], "claim": c.text,
                   "claim_gold": int(overlaps(c, ex["labels"])),
                   "resp_gold": int(len(ex["labels"]) > 0)}
            for name in dets:
                row[name] = per_method[name][k]
                row[f"{name}_ms"] = times[name]
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(dev, test, methods):
    out = []
    for level in ["response", "claim"]:
        for m in methods:
            if level == "response":
                d = dev.groupby("id").agg(s=(m, "max"), y=("resp_gold", "first"))
                t = test.groupby("id").agg(s=(m, "max"), y=("resp_gold", "first"),
                                           ms=(f"{m}_ms", "first"))
            else:
                d = dev.rename(columns={m: "s", "claim_gold": "y"})
                t = test.rename(columns={m: "s", "claim_gold": "y", f"{m}_ms": "ms"})
            thr = best_threshold(d["s"].values, d["y"].values)
            p, r, f1, _ = precision_recall_fscore_support(t["y"], t["s"] >= thr, average="binary",
                                                          zero_division=0)
            auroc = roc_auc_score(t["y"], t["s"]) if t["y"].nunique() > 1 else float("nan")
            out.append({"level": level, "method": m, "threshold": thr, "precision": round(p, 3),
                        "recall": round(r, 3), "f1": round(f1, 3), "auroc": round(auroc, 3),
                        "ms_per_answer": round(test.groupby("id")[f"{m}_ms"].first().mean(), 1)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--methods", default="nli")
    ap.add_argument("--task", default="all", choices=["all", "QA", "Summary", "Data2txt"])
    ap.add_argument("--n_dev", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    methods = args.methods.split(",")

    responses = load_jsonl(os.path.join(args.data_dir, "response.jsonl"))
    sources = {s["source_id"]: s for s in load_jsonl(os.path.join(args.data_dir, "source_info.jsonl"))}

    by_split = defaultdict(list)
    for r in responses:
        src = sources[r["source_id"]]
        if args.task != "all" and src["task_type"] != args.task:
            continue
        by_split[r["split"]].append({
            "id": r["id"], "task": src["task_type"], "response": r["response"],
            "labels": r.get("labels", []), "context": get_context(src),
            "prompt": src.get("prompt") or get_context(src),
        })

    rng = random.Random(args.seed)
    dev = rng.sample(by_split["train"], min(args.n_dev, len(by_split["train"])))
    test = rng.sample(by_split["test"], min(args.n_test, len(by_split["test"])))
    print(f"dev={len(dev)} test={len(test)} methods={methods}")

    dets = build_detectors(methods)
    dev_df, test_df = run(dev, dets), run(test, dets)

    os.makedirs(args.out, exist_ok=True)
    test_df.to_csv(os.path.join(args.out, "test_claim_scores.csv"), index=False)
    table = evaluate(dev_df, test_df, methods)
    table.to_csv(os.path.join(args.out, f"summary_{'_'.join(methods)}.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
