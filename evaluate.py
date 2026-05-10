"""
ReviewLens - Evaluation harness.

Runs the classifier on a labeled test set and computes:
- Accuracy, precision, recall, F1
- Confusion matrix
- Per-sample predictions
- Comparison: with RAG vs without RAG
"""

import gc
import json
import re
import time
from pathlib import Path

import ollama
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tabulate import tabulate

from app.pipeline import (
    retrieve_similar,
    classify_review,
    TEXT_MODEL,
)


TEST_SET_PATH = Path("data/test_set.json")
RESULTS_PATH = Path("data/evaluation_results.json")


# ---------- Baseline classifier (no RAG) ----------
NO_RAG_PROMPT = """You are an expert review authenticity analyst. Classify a new review as Real, Fake, or Suspicious.

CLASSIFICATION FRAMEWORK:
- REAL: specific details, personal context, balanced opinions, mentions of actual use
- FAKE: generic praise, no specifics, marketing-speak, repetitive superlatives
- SUSPICIOUS: too short/vague to confidently judge, mixed signals

NEW REVIEW TO CLASSIFY:
{review_json}

Output ONLY valid JSON:

{{
  "verdict": "Real" | "Fake" | "Suspicious",
  "confidence": float 0.0 to 1.0,
  "reasoning": "1-2 sentences"
}}

JSON only:"""


def classify_no_rag(review_data):
    """Baseline classifier without RAG retrieval."""
    review_json_str = json.dumps(review_data, indent=2)
    prompt = NO_RAG_PROMPT.format(review_json=review_json_str)
    
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_ctx": 2048},
    )
    
    raw = response["message"]["content"].strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"verdict": "Unknown", "confidence": 0.0}


# ---------- Run evaluation ----------
def evaluate(test_set, mode="rag"):
    """Run classifier on each test sample. mode = 'rag' or 'baseline'."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {mode.upper()}")
    print(f"{'='*60}")
    
    predictions = []
    truths = []
    rows = []
    
    for i, sample in enumerate(test_set, 1):
        print(f"\n[{i}/{len(test_set)}] {sample['id']} (true: {sample['true_label']})")
        
        # Build a structured-review dict so the classifier prompt has consistent input
        review_data = {
            "review_text": sample["review_text"],
            "verified_purchase": sample["verified_purchase"],
            "platform": sample["platform"],
        }
        
        start = time.time()
        
        if mode == "rag":
            examples = retrieve_similar(review_data["review_text"])
            verdict = classify_review(review_data, examples)
        else:
            verdict = classify_no_rag(review_data)
        
        elapsed = time.time() - start
        predicted = verdict.get("verdict", "Unknown")
        confidence = verdict.get("confidence", 0)
        
        print(f"   Predicted: {predicted} ({confidence:.0%}) — {elapsed:.1f}s")
        
        predictions.append(predicted)
        truths.append(sample["true_label"])
        rows.append({
            "id": sample["id"],
            "true": sample["true_label"],
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == sample["true_label"],
            "elapsed_sec": round(elapsed, 1),
        })
        
        gc.collect()
    
    return predictions, truths, rows


# ---------- Compute metrics ----------
def report_metrics(truths, predictions, mode):
    print(f"\n{'='*60}")
    print(f"METRICS: {mode.upper()}")
    print(f"{'='*60}")
    
    labels = ["Real", "Fake", "Suspicious"]
    
    # Filter to known labels (drop "Unknown" predictions for metric calculation)
    valid_pairs = [(t, p) for t, p in zip(truths, predictions) if p in labels]
    if not valid_pairs:
        print("No valid predictions to evaluate.")
        return {}
    
    valid_truths, valid_preds = zip(*valid_pairs)
    
    accuracy = accuracy_score(valid_truths, valid_preds)
    print(f"\nAccuracy: {accuracy:.1%} ({sum(t == p for t, p in valid_pairs)}/{len(valid_pairs)} correct)")
    
    print("\nClassification Report:")
    report = classification_report(
        valid_truths, valid_preds, labels=labels, zero_division=0, digits=3
    )
    print(report)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(valid_truths, valid_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(tabulate(cm_df, headers="keys", tablefmt="grid"))
    
    return {
        "accuracy": accuracy,
        "predictions": list(valid_preds),
        "truths": list(valid_truths),
        "confusion_matrix": cm.tolist(),
    }


# ---------- Main ----------
def main():
    print("Loading test set...")
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"Loaded {len(test_set)} test samples")
    
    # Evaluate baseline (no RAG)
    base_preds, base_truths, base_rows = evaluate(test_set, mode="baseline")
    base_metrics = report_metrics(base_truths, base_preds, "baseline")
    
    # Evaluate RAG-enhanced
    rag_preds, rag_truths, rag_rows = evaluate(test_set, mode="rag")
    rag_metrics = report_metrics(rag_truths, rag_preds, "rag")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: BASELINE vs RAG")
    print(f"{'='*60}")
    
    comparison = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Baseline (no RAG)": [f"{base_metrics.get('accuracy', 0):.1%}"],
        "With RAG": [f"{rag_metrics.get('accuracy', 0):.1%}"],
    })
    print(tabulate(comparison, headers="keys", tablefmt="grid", showindex=False))
    
    # Per-sample table
    print("\nPer-Sample Predictions (RAG):")
    df = pd.DataFrame(rag_rows)
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    
    # Save results
    results = {
        "test_set_size": len(test_set),
        "baseline": {
            "accuracy": base_metrics.get("accuracy"),
            "rows": base_rows,
        },
        "rag": {
            "accuracy": rag_metrics.get("accuracy"),
            "rows": rag_rows,
        },
    }
    
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()