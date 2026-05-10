"""
ReviewLens core pipeline — used by both CLI scripts and the Streamlit UI.
"""

import gc
import json
import re
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ---------- Configuration ----------
TEXT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "reviews"
TOP_K = 3


# ---------- OCR ----------
def run_ocr(image_path: str | Path) -> str:
    """Extract text from an image using EasyOCR."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(str(image_path))
    lines = [text for _, text, conf in results if conf > 0.5]
    text = "\n".join(lines)
    del reader, results
    gc.collect()
    return text


# ---------- Structuring ----------
STRUCTURING_PROMPT = """Extract structured data from this OCR text of a product review.

OCR text:
---
{ocr_text}
---

PLATFORM: "Reviewed in [country] on..." = Amazon | "Certified Buyer" = Flipkart | "Local Guide" = Google | else "unknown"

Output ONLY valid JSON:

{{
  "reviewer_name": "string or null",
  "review_title": "string or null",
  "review_text": "main body only, exclude title/date/metadata",
  "review_date": "string or null",
  "verified_purchase": true or false,
  "platform": "Amazon | Flipkart | Google | Yelp | unknown",
  "helpful_votes": number or null,
  "product_details": "string or null"
}}

JSON only:"""


def structure_review(ocr_text: str) -> dict:
    """Convert raw OCR text to structured JSON via Llama."""
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": STRUCTURING_PROMPT.format(ocr_text=ocr_text)}],
        options={"temperature": 0.1, "num_ctx": 2048},
    )
    raw = response["message"]["content"].strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ---------- Retrieval ----------
def retrieve_similar(review_text: str, top_k: int = TOP_K) -> list[dict]:
    """Find top-k similar labeled reviews from ChromaDB."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = embedder.encode([review_text]).tolist()
    
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    
    examples = []
    for i in range(len(results["ids"][0])):
        examples.append({
            "label": results["metadatas"][0][i]["label"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    
    del embedder
    gc.collect()
    return examples


# ---------- Classification ----------
RAG_CLASSIFICATION_PROMPT = """You are an expert review authenticity analyst. Classify a new review as Real, Fake, or Suspicious.

CLASSIFICATION FRAMEWORK:
- REAL: specific details, personal context, balanced opinions, mentions of actual use
- FAKE: generic praise, no specifics, marketing-speak, repetitive superlatives
- SUSPICIOUS: too short/vague to confidently judge, mixed signals

REFERENCE EXAMPLES (similar reviews from a labeled dataset):
{examples_block}

NEW REVIEW TO CLASSIFY:
{review_json}

ANALYSIS STEPS:
1. Compare to reference examples — which label does it most resemble?
2. Specificity: real product details vs generic praise?
3. Language: personal experience vs marketing-speak?
4. Consider verified_purchase and helpful_votes.

Output ONLY valid JSON (no markdown):

{{
  "verdict": "Real" | "Fake" | "Suspicious",
  "confidence": float 0.0 to 1.0,
  "red_flags": ["specific concerns found"],
  "green_flags": ["specific authenticity signals found"],
  "similar_to": "Real | Fake | Suspicious",
  "reasoning": "2-3 sentences explaining the verdict"
}}

JSON only:"""


def format_examples(examples: list[dict]) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i} [Label: {ex['label']}, distance: {ex['distance']:.2f}]:\n"
            f"\"{ex['text']}\""
        )
    return "\n\n".join(blocks)


def classify_review(review_data: dict, examples: list[dict]) -> dict:
    """Classify using RAG-enhanced few-shot prompt."""
    review_json_str = json.dumps(review_data, indent=2, ensure_ascii=False)
    examples_block = format_examples(examples)
    
    prompt = RAG_CLASSIFICATION_PROMPT.format(
        examples_block=examples_block,
        review_json=review_json_str,
    )
    
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    
    raw = response["message"]["content"].strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"_raw_output": raw, "_parse_error": str(e)}


# ---------- Full pipeline ----------
def analyze_screenshot(image_path: str | Path) -> dict:
    """Run the full pipeline and return all intermediate + final results."""
    ocr_text = run_ocr(image_path)
    structured = structure_review(ocr_text)
    review_text = structured.get("review_text", "") if structured else ""
    examples = retrieve_similar(review_text) if review_text else []
    verdict = classify_review(structured, examples) if structured else {}
    
    return {
        "ocr_text": ocr_text,
        "structured": structured,
        "examples": examples,
        "verdict": verdict,
    }