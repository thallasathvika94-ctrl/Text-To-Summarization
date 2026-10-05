"""
Text Summarization App
Summarizes long text, .txt or .pdf files using BART models (abstractive
summarization). Long documents are split into chunks, summarized, and merged.
Runs locally. No API key needed.
"""
import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")

MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}

LENGTHS = {  # (min_tokens, max_tokens) per chunk
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}

CHUNK_WORDS = 600  # BART accepts ~1024 tokens; 600 words keeps us safely under


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


def chunk_text(text: str, size: int = CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def summarize_chunk(text, tokenizer, model, min_len, max_len):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize(text, tokenizer, model, min_len, max_len):
    chunks = chunk_text(text)
    progress = st.progress(0.0, text="Summarizing...")
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Very short chunks can't hit min_len; relax it
        words = len(chunk.split())
        parts.append(summarize_chunk(chunk, tokenizer, model, min(min_len, words // 2), max_len))
        progress.progress(i / len(chunks), text=f"Summarized chunk {i}/{len(chunks)}")
    progress.empty()

    combined = " ".join(parts)
    # If the merged summary is still long, summarize it once more
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:
        combined = summarize_chunk(combined, tokenizer, model, min_len, max_len * 2)
    return combined


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
length = st.sidebar.radio("Summary length", list(LENGTHS.keys()), index=1)

# ---------------- Main ----------------
st.title("📝 Text Summarizer")
st.write("Paste text or upload a file (.txt / .pdf) to get a concise summary.")

source = st.radio("Input type", ["Paste text", "Upload file"], horizontal=True)
text = ""
if source == "Paste text":
    text = st.text_area("Your text", height=300, placeholder="Paste an article, notes, or report here...")
else:
    file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
    if file:
        text = read_pdf(file) if file.name.lower().endswith(".pdf") else file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview extracted text"):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

if st.button("⚡ Summarize", type="primary"):
    word_count = len(text.split())
    if word_count < 50:
        st.warning("Please provide at least 50 words of text.")
    else:
        tokenizer, model = load_model(MODELS[model_name])
        min_len, max_len = LENGTHS[length]
        summary = summarize(text, tokenizer, model, min_len, max_len)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.caption(f"{word_count} words")
            st.write(text[:5000] + ("..." if len(text) > 5000 else ""))
        with col2:
            st.subheader("Summary")
            s_count = len(summary.split())
            st.caption(f"{s_count} words • {100 - s_count * 100 // word_count}% shorter")
            st.success(summary)
            st.download_button("⬇️ Download summary", summary, "summary.txt", "text/plain")
