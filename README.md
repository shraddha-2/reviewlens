# 🔍 ReviewLens

**AI-powered fake review detection from screenshots.** Upload any product review image and get an instant verdict — Real, Fake, or Suspicious — with reasoning.

Runs **100% locally** with open-source models. No API keys, no data sent to the cloud.

---

## 🎯 Demo

![ReviewLens Demo](docs/demo.gif)

> Drag a screenshot in → wait ~30 seconds → get a verdict with red/green flags and reasoning.

---

## 📊 Results

Evaluated on a held-out test set of 9 labeled reviews:

| Approach | Accuracy |
|---|---|
| Baseline (LLM only) | 66.7% |
| **+ RAG retrieval** | **88.9% (+22.2pp)** |

Per-class breakdown (RAG):
- **Real reviews:** 3/3 correct (100%)
- **Fake reviews:** 3/3 correct (100%)
- **Suspicious reviews:** 2/3 correct (67%)

See [`data/evaluation_results.json`](data/evaluation_results.json) for full results.

---

## 🏗️ Architecture
Screenshot → OCR → Structured Data → Vector Retrieval → RAG-Enhanced Classification → Verdict
📸        🔤         📋                🔍                    🤖                     ✅

Pipeline stages:

1. **OCR (EasyOCR)** — extracts raw text from the screenshot
2. **Structured Extraction (Llama 3.2)** — parses OCR text into reviewer name, title, body, date, verified purchase, helpful votes, etc.
3. **Vector Retrieval (ChromaDB + sentence-transformers)** — finds the top-3 most similar labeled reviews from a reference dataset
4. **RAG-Enhanced Classification (Llama 3.2)** — classifies as Real / Fake / Suspicious using retrieved examples as few-shot context
5. **Output** — verdict with confidence, red/green flags, and reasoning grounded in the retrieved examples

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| OCR | EasyOCR |
| LLM | Llama 3.2 (3B) via Ollama |
| Vector DB | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| UI | Streamlit |
| Evaluation | scikit-learn, pandas |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- ~8 GB free RAM
- ~10 GB disk space (for models)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/reviewlens.git
cd reviewlens

# Create virtual environment (using uv — fastest)
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
uv pip install -r requirements.txt

# Pull the required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Build the vector database (one-time setup)
python build_vector_db.py
```

### Run the Web UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` and drag in a review screenshot.

### Run the Evaluation

```bash
python evaluate.py
```

---

## 📁 Project Structure
reviewlens/
├── app/
│   ├── init.py
│   └── pipeline.py          # Core OCR + LLM + RAG pipeline
├── data/
│   ├── reference_reviews.json    # Labeled examples for RAG
│   ├── test_set.json             # Held-out evaluation set
│   └── evaluation_results.json   # Latest benchmark results
├── sample_screenshots/      # Test screenshots
├── streamlit_app.py         # Web UI
├── build_vector_db.py       # One-time DB setup
├── evaluate.py              # Run evaluation harness
├── requirements.txt
└── README.md

---

## 🧠 Design Decisions

**Why RAG over fine-tuning?**
Fine-tuning a 3B model would require labeled data, GPU, and time. RAG with a small reference dataset achieves +22pp accuracy with zero training and is easy to update — just add new labeled examples to `reference_reviews.json` and re-run `build_vector_db.py`.

**Why Llama 3.2 3B over larger models?**
Runs comfortably on CPU with 8 GB RAM. Latency is ~30s per classification — acceptable for an interactive tool. Larger models would improve accuracy but require GPU.

**Why EasyOCR over PaddleOCR or LLaVA?**
LLaVA hallucinated review content during testing. PaddleOCR has known compatibility issues on Windows CPU. EasyOCR is reliable, accurate (~98% confidence on clean screenshots), and Windows-friendly.

---

## 🔬 Limitations

- Test set is small (9 samples). Numbers will shift with more data.
- Reference dataset is hand-crafted (15 examples). A scraped dataset would improve generalization.
- Latency is ~30-50s per review on CPU. Production deployment would benefit from GPU or batching.
- Suspicious reviews are the hardest class — short verified-purchase reviews are genuinely ambiguous.

---

## 🛣️ Roadmap

- [ ] Scale reference dataset to 200+ examples (using Amazon Fake Reviews dataset)
- [ ] Add batch processing for evaluating dozens of screenshots
- [ ] Fine-tune a small model on the reference set and compare to RAG
- [ ] Deploy to Hugging Face Spaces / Railway with hosted LLM
- [ ] Add a confidence calibration step

---

## 📜 License

MIT License — feel free to use, modify, and learn from this project.

---

## 👤 Author

**Shraddha** — Machine Learning Engineer

Built as a portfolio project demonstrating end-to-end ML engineering: OCR, LLMs, RAG, evaluation, and deployment.