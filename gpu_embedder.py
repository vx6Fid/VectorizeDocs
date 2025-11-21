from sentence_transformers import SentenceTransformer

# Load model once at module import time (this happens per process/thread).
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    # Expect chunks: list of dicts with "text"
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=16, convert_to_numpy=True)
    return embeddings
