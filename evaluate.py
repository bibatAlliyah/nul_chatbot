# Evaluation for the NU Laguna chatbot.
#
#   python evaluate.py                 -> evaluate on test_data.csv (held-out questions)
#   python evaluate.py --mode loo      -> leave-one-out over the training patterns
#   python evaluate.py --mode both
#   python evaluate.py --export-csv    -> rewrite training_data.csv from intents_merged.json
#
# Results are printed and saved to eval_output/:
#   <mode>_metrics.json, <mode>_per_intent.csv, <mode>_confusion_matrix.csv,
#   <mode>_errors.csv, and <mode>_confusion_matrix.png (if matplotlib is installed).
# The web app shows the same results at http://127.0.0.1:5000/eval
#
# Two ways to test, and why both exist:
#   test  - questions the bot has NEVER seen (test_data.csv). This is the
#           honest number to report. Add more rows (ask classmates to write
#           questions the way they'd really type them) to make it stronger.
#   loo   - "leave-one-out": each training pattern is hidden from the bot and
#           classified using all the OTHER patterns. Uses data you already
#           have, but patterns are often near-copies of each other, so it
#           usually scores a bit higher than the real test set.
#
# Testing the bot on the exact patterns it matches against would score ~100%
# and prove nothing (each question would just find itself), which is why
# neither mode does that.

import argparse
import csv
import json
import os

import numpy as np

import chatbot
from sentence_transformers import util

TEST_PATH = "test_data.csv"
OUTPUT_DIR = "eval_output"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def predict_rows(texts, true_intents, similarity_matrix, res, cleaned_texts, corrections_list):
    """One prediction per row. similarity_matrix[i] holds row i's cosine
    similarity to every training pattern."""
    rows = []
    for i, (text, true_intent) in enumerate(zip(texts, true_intents)):
        ranked = chatbot.rank_from_similarities(similarity_matrix[i], res.pattern_intents, res.pattern_raw)
        best, score, second, gap = chatbot.summarize_ranking(ranked)
        decision, _ = chatbot.decide(best, score, second, gap, bool(res.departments))
        rows.append({
            "text": text,
            "normalized": cleaned_texts[i],
            "corrections": corrections_list[i],
            "true": true_intent,
            "predicted": best,
            "score": round(score, 4),
            "second": second,
            "gap": round(gap, 4),
            "decision": decision,
            "matched_pattern": ranked[0]["pattern"],
        })
    return rows


def load_test_set(path, res):
    texts, labels, skipped = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text, intent = row["text"].strip(), row["intent"].strip()
            if not text:
                continue
            if intent not in res.response_lookup:
                skipped.append(intent)
                continue
            texts.append(text)
            labels.append(intent)
    return texts, labels, skipped


def predict_test_set(res, path=TEST_PATH):
    texts, labels, skipped = load_test_set(path, res)
    if not texts:
        raise ValueError(f"No usable rows in {path}")
    processed = [chatbot.preprocess(t, res) for t in texts]
    cleaned = [p[0] for p in processed]
    corrections = [p[1] for p in processed]
    embeddings = res.embedder.encode(cleaned, convert_to_tensor=True)
    sims = chatbot.to_numpy(util.cos_sim(embeddings, res.pattern_embeddings))
    rows = predict_rows(texts, labels, sims, res, cleaned, corrections)
    note = f"{len(rows)} held-out questions from {path}"
    if skipped:
        note += f" ({len(skipped)} rows skipped: unknown intents {sorted(set(skipped))})"
    return rows, note


def predict_leave_one_out(res):
    emb = chatbot.to_numpy(res.pattern_embeddings).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    sims = emb @ emb.T
    texts = np.array(res.pattern_texts)
    for i in range(len(texts)):
        # Hide the pattern itself AND any identical copy of it, otherwise a
        # duplicated pattern would just find its twin.
        sims[i, texts == texts[i]] = -1.0
    rows = predict_rows(res.pattern_raw, res.pattern_intents, sims, res,
                        res.pattern_texts, [[] for _ in texts])
    return rows, f"{len(rows)} training patterns, each hidden in turn"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(rows, intent_order):
    """Accuracy, per-intent precision/recall/F1, macro & weighted averages,
    confusion matrix, and how the thresholds behave."""
    true = [r["true"] for r in rows]
    pred = [r["predicted"] for r in rows]
    present = set(true) | set(pred)
    labels = [t for t in intent_order if t in present]
    index = {t: i for i, t in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)  # rows = true, cols = predicted
    for t, p in zip(true, pred):
        matrix[index[t], index[p]] += 1

    per_intent = []
    for t in labels:
        i = index[t]
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        support = int(matrix[i, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent.append({"intent": t, "precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(f1, 4), "support": support})

    with_support = [m for m in per_intent if m["support"] > 0]
    total = sum(m["support"] for m in with_support)

    def macro(key):
        return round(sum(m[key] for m in with_support) / len(with_support), 4)

    def weighted(key):
        return round(sum(m[key] * m["support"] for m in with_support) / total, 4)

    correct = sum(t == p for t, p in zip(true, pred))

    # How the bot ACTUALLY behaves with its thresholds.
    answered = [r for r in rows if r["decision"] == "answer"]
    dym = [r for r in rows if r["decision"] == "did_you_mean"]
    fallback = [r for r in rows if r["decision"] == "fallback"]
    dym_hit = [r for r in dym if r["true"] in (r["predicted"], r["second"])]
    n = len(rows)

    confused = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if i != j and matrix[i, j]:
                confused.append({"true": t, "predicted": p, "count": int(matrix[i, j])})
    confused.sort(key=lambda c: -c["count"])

    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "macro": {"precision": macro("precision"), "recall": macro("recall"), "f1": macro("f1")},
        "weighted": {"precision": weighted("precision"), "recall": weighted("recall"), "f1": weighted("f1")},
        "per_intent": per_intent,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "most_confused": confused[:15],
        "behaviour": {
            "answered_rate": round(len(answered) / n, 4),
            "answered_accuracy": round(sum(r["true"] == r["predicted"] for r in answered) / len(answered), 4) if answered else None,
            "did_you_mean_rate": round(len(dym) / n, 4),
            "did_you_mean_contains_correct": round(len(dym_hit) / len(dym), 4) if dym else None,
            "fallback_rate": round(len(fallback) / n, 4),
            "fallback_but_top1_correct": sum(r["true"] == r["predicted"] for r in fallback),
        },
        "thresholds": {"similarity": chatbot.SIMILARITY_THRESHOLD, "min_gap": chatbot.MIN_GAP,
                       "near_miss_gap": chatbot.NEAR_MISS_GAP},
        "errors": [r for r in rows if r["true"] != r["predicted"]],
    }


def evaluate(res, mode="test", test_path=TEST_PATH):
    """Returns a JSON-friendly dict with every metric. Used by the CLI and by /eval."""
    if mode == "test":
        rows, note = predict_test_set(res, test_path)
    elif mode == "loo":
        rows, note = predict_leave_one_out(res)
    else:
        raise ValueError("mode must be 'test' or 'loo'")
    intent_order = [tag for tag, _ in res.guide]
    result = compute_metrics(rows, intent_order)
    result["mode"] = mode
    result["note"] = note
    result["model"] = chatbot.MODEL_NAME
    return result


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def save_outputs(result, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    mode = result["mode"]

    summary = {k: v for k, v in result.items() if k not in ("errors", "confusion_matrix")}
    with open(os.path.join(out_dir, f"{mode}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, f"{mode}_per_intent.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent", "precision", "recall", "f1", "support"])
        w.writeheader()
        w.writerows(result["per_intent"])

    with open(os.path.join(out_dir, f"{mode}_confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + result["labels"])
        for label, row in zip(result["labels"], result["confusion_matrix"]):
            w.writerow([label] + row)

    with open(os.path.join(out_dir, f"{mode}_errors.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["text", "normalized", "true", "predicted", "score", "second", "gap", "decision", "matched_pattern"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(result["errors"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels, m = result["labels"], np.array(result["confusion_matrix"])
        size = max(8, len(labels) * 0.32)
        fig, ax = plt.subplots(figsize=(size, size))
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Predicted intent")
        ax.set_ylabel("True intent")
        ax.set_title(f"Confusion matrix ({mode}) - accuracy {result['accuracy']:.1%}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=6,
                            color="white" if m[i, j] > m.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{mode}_confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("(matplotlib not installed: skipped the confusion matrix PNG. "
              "pip install matplotlib to get it.)")


def print_report(result):
    print(f"\n=== Evaluation: {result['mode']} ({result['note']}) ===")
    print(f"Accuracy (top-1):   {result['accuracy']:.2%}")
    for avg in ("macro", "weighted"):
        a = result[avg]
        print(f"{avg.capitalize():9} avg:      precision {a['precision']:.3f}  recall {a['recall']:.3f}  F1 {a['f1']:.3f}")
    b = result["behaviour"]
    print(f"\nWith thresholds (similarity {chatbot.SIMILARITY_THRESHOLD}, gap {chatbot.MIN_GAP}/{chatbot.NEAR_MISS_GAP}):")
    answered_acc = "-" if b["answered_accuracy"] is None else f"{b['answered_accuracy']:.1%}"
    print(f"  answered directly: {b['answered_rate']:.1%}  (correct when answered: {answered_acc})")
    print(f"  did you mean:      {b['did_you_mean_rate']:.1%}")
    print(f"  not understood:    {b['fallback_rate']:.1%}")

    print(f"\n{'intent':28} {'prec':>6} {'recall':>6} {'F1':>6} {'n':>4}")
    for m in sorted(result["per_intent"], key=lambda m: m["f1"]):
        if m["support"]:
            print(f"{m['intent']:28} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f} {m['support']:4}")

    if result["most_confused"]:
        print("\nMost confused (true -> predicted):")
        for c in result["most_confused"][:8]:
            print(f"  {c['true']} -> {c['predicted']}  x{c['count']}")
    print(f"\n{len(result['errors'])} misclassified rows saved to {OUTPUT_DIR}/{result['mode']}_errors.csv")


def export_training_csv(res, path="training_data.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent"])
        w.writerows(zip(res.pattern_raw, res.pattern_intents))
    print(f"Wrote {len(res.pattern_raw)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the NU Laguna chatbot.")
    parser.add_argument("--mode", choices=["test", "loo", "both"], default="test")
    parser.add_argument("--test-file", default=TEST_PATH)
    parser.add_argument("--export-csv", action="store_true", help="rewrite training_data.csv and exit")
    args = parser.parse_args()

    res = chatbot.load_resources(chatbot.DATA_PATH)
    if args.export_csv:
        export_training_csv(res)
        return

    for mode in (["test", "loo"] if args.mode == "both" else [args.mode]):
        result = evaluate(res, mode, args.test_file)
        print_report(result)
        save_outputs(result)


if __name__ == "__main__":
    main()
