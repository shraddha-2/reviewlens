"""
Build the vector database of labeled reviews.
Run this ONCE — it creates a persistent ChromaDB at ./chroma_db
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

REFERENCE_FILE = Path("data/reference_reviews.json")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "reviews"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80 MB, fast on CPU


def main():
    print("Loading reference reviews...")
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        reviews = json.load(f)
    print(f"Loaded {len(reviews)} reference reviews")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    texts = [r["review_text"] for r in reviews]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Setting up ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete old collection if it exists, so we can re-run cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Inserting reviews into vector DB...")
    collection.add(
        ids=[r["id"] for r in reviews],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "label": r["label"],
                "verified_purchase": r["verified_purchase"],
                "rating": r["rating"],
            }
            for r in reviews
        ],
    )

    print(f"\nVector DB built successfully with {collection.count()} reviews.")
    print("You only need to run this once (or whenever you update the reference set).")


if __name__ == "__main__":
    main()