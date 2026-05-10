"""
ReviewLens - Streamlit Web UI
Run with: streamlit run streamlit_app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from app.pipeline import (
    run_ocr,
    structure_review,
    retrieve_similar,
    classify_review,
)


# ---------- Page config ----------
st.set_page_config(
    page_title="ReviewLens — Fake Review Detector",
    page_icon="🔍",
    layout="wide",
)


# ---------- Header ----------
st.title("🔍 ReviewLens")
st.caption(
    "Upload a screenshot of any product review and get an AI-powered authenticity verdict — "
    "Real, Fake, or Suspicious — with reasoning."
)
st.divider()


# ---------- Sidebar info ----------
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **OCR** extracts text from your screenshot
    2. **Llama 3.2** structures the review data
    3. **Vector search** finds similar labeled reviews
    4. **RAG-enhanced classifier** delivers a verdict
    """)
    st.divider()
    st.subheader("Tech Stack")
    st.markdown("""
    - EasyOCR for text extraction
    - Llama 3.2 (3B) via Ollama
    - ChromaDB for vector storage
    - Sentence-Transformers for embeddings
    - Streamlit for this UI
    """)
    st.divider()
    st.caption("Runs 100% locally. No data leaves your machine.")


# ---------- Main area ----------
uploaded_file = st.file_uploader(
    "Upload review screenshot",
    type=["png", "jpg", "jpeg"],
    help="Drag and drop or click to select. Works with Amazon, Flipkart, Google, Yelp screenshots.",
)


if uploaded_file is not None:
    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Screenshot")
        image = Image.open(tmp_path)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔬 Analysis")
        
        # Run the pipeline with progress indicators
        progress_text = st.empty()
        
        with st.spinner("Running OCR..."):
            progress_text.text("Step 1/4: Extracting text...")
            ocr_text = run_ocr(tmp_path)
        
        with st.spinner("Structuring data..."):
            progress_text.text("Step 2/4: Structuring with Llama 3.2...")
            structured = structure_review(ocr_text)
        
        if not structured or not structured.get("review_text"):
            st.error("Could not extract review data from this screenshot. Try a clearer image.")
            st.stop()
        
        with st.spinner("Retrieving similar examples..."):
            progress_text.text("Step 3/4: Finding similar labeled reviews...")
            examples = retrieve_similar(structured.get("review_text", ""))
        
        with st.spinner("Classifying..."):
            progress_text.text("Step 4/4: Generating verdict...")
            verdict = classify_review(structured, examples)
        
        progress_text.empty()
        
        # Verdict header
        v = verdict.get("verdict", "Unknown")
        conf = verdict.get("confidence", 0)
        
        verdict_colors = {
            "Real": "🟢",
            "Fake": "🔴",
            "Suspicious": "🟡",
        }
        icon = verdict_colors.get(v, "⚪")
        
        st.markdown(f"### {icon} Verdict: **{v}**")
        st.progress(float(conf), text=f"Confidence: {conf:.0%}")
        
        # Flags
        col_red, col_green = st.columns(2)
        with col_red:
            st.markdown("**🚩 Red Flags**")
            red_flags = verdict.get("red_flags", []) or ["(none detected)"]
            for flag in red_flags:
                st.markdown(f"- {flag}")
        
        with col_green:
            st.markdown("**✅ Green Flags**")
            green_flags = verdict.get("green_flags", []) or ["(none detected)"]
            for flag in green_flags:
                st.markdown(f"- {flag}")
        
        # Reasoning
        st.markdown("**💭 Reasoning**")
        st.info(verdict.get("reasoning", "No reasoning provided."))
    
    # Full-width sections below
    st.divider()
    
    # Extracted data (expandable)
    with st.expander("📋 View extracted review data"):
        st.json(structured)
    
    # Retrieved examples (expandable)
    with st.expander(f"🔍 View {len(examples)} retrieved similar reviews"):
        for i, ex in enumerate(examples, 1):
            label = ex["label"]
            label_icon = verdict_colors.get(label, "⚪")
            st.markdown(f"**Example {i}** {label_icon} `{label}` (distance: {ex['distance']:.2f})")
            st.markdown(f"> {ex['text']}")
            st.markdown("")
    
    # Raw OCR (expandable)
    with st.expander("📝 View raw OCR text"):
        st.text(ocr_text)

else:
    st.info("👆 Upload a review screenshot above to get started.")
    
    # Show a tip section when no file is uploaded
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        - Capture the **full review** including reviewer name, rating, and date
        - Use **PNG or JPEG** format
        - Ensure text is **clear and readable** (no blur)
        - Works best with reviews from **Amazon, Flipkart, Google, Yelp**
        - First analysis takes ~45 seconds (loading models). Subsequent ones are faster.
        """)