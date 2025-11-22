import logging
import traceback
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

# Load model once at module import time (this happens per process/thread).

# Configure logging with timestamps and level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tender_processor")

model = None
# If you also use SentenceTransformer locally, keep this load (will run at import)
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("SentenceTransformer model loaded at module import time.")
except Exception:
    model = None
    logger.exception("Failed to load SentenceTransformer model at import time.")


# def embed_chunks(chunks):
#     # Expect chunks: list of dicts with "text"
#     texts = [c["text"] for c in chunks]
#     embeddings = model.encode(texts, batch_size=16, convert_to_numpy=True)
#     return embeddings


def embed_batch(chunks):
    if model is None:
        raise ValueError("SentenceTransformer model is not loaded.")
    texts = [c["data"] for c in chunks]
    vectors = model.encode(texts, batch_size=16, show_progress_bar=False).tolist()

    out = []
    for c, emb in zip(chunks, vectors):
        out.append(
            {
                "tender_id": c["tender_id"],
                "document_name": c["document_name"],
                "page": c["page"],
                "position": c["position"],
                "sub_position": c["sub_position"],
                "type": c["type"],
                "is_scanned": c["is_scanned"],
                "text": c["data"],
                "embedding": emb,
            }
        )

    return out


# Local embed_chunks implementation with heavy instrumentation:
def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Embeds chunks. This local function uses SentenceTransformer model (if loaded).
    It will try to be tolerant to chunk schema differences ('text' vs 'data') and will
    print debugging info about chunk shapes and sample text lengths.
    Returns list of dicts containing metadata and embedding.
    """
    logger.debug("embed_chunks called with %d chunks", len(chunks))
    if not chunks:
        logger.warning("embed_chunks called with empty chunks list")
        return []

    # Inspect keys from first few chunks
    try:
        sample_keys = [list(c.keys()) for c in chunks[:3]]
        logger.debug("Sample chunk keys (first 3): %s", sample_keys)
    except Exception:
        logger.exception("Failed to read chunk keys")

    # Build texts robustly
    texts = []
    for i, c in enumerate(chunks):
        if "text" in c and isinstance(c["text"], str):
            texts.append(c["text"])
        elif "data" in c and isinstance(c["data"], str):
            texts.append(c["data"])
        else:
            # fallback, try to stringify
            try:
                texts.append(str(c.get("text") or c.get("data") or ""))
            except Exception:
                texts.append("")
                logger.warning(
                    "Chunk %d had no text/data fields; inserted empty string", i
                )

    # Log basic stats about texts
    try:
        lengths = [len(t) for t in texts]
        logger.debug("Text lengths (first 10): %s", lengths[:10])
        logger.info(
            "Embedding %d texts (min_len=%d, max_len=%d)",
            len(texts),
            min(lengths),
            max(lengths),
        )
    except Exception:
        logger.exception("Failed to compute text lengths")

    # Use model if available; if you have an external embedder function, call it instead
    if model is None:
        # If there's an external GPU embedder provided, try it
        try:
            logger.warning(
                "Local SentenceTransformer model not available; attempting external embedder."
            )
            vectors = embed_batch(chunks)  # might throw if not set up
            logger.info("external embedder returned %d vectors", len(vectors))
        except Exception:
            tb = traceback.format_exc()
            logger.exception("External GPU embedder not available or failed: %s", tb)
            raise RuntimeError(
                "No embedding method available. Load SentenceTransformer or provide GPU embedder."
            )
    else:
        try:
            logger.debug("Encoding using SentenceTransformer.model.encode")
            vectors = model.encode(texts, batch_size=16, show_progress_bar=False)
            logger.info(
                "Local model.encode returned shape-like object: %s", type(vectors)
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("model.encode failed: %s", e)
            raise

    # Package output consistent with the rest of the code
    out = []
    try:
        # If vectors is a list of lists or numpy array-like, make sure to iterate properly
        for c, emb in zip(chunks, vectors):
            out.append(
                {
                    "tender_id": c.get("tender_id"),
                    "document_name": c.get("document_name"),
                    "page": c.get("page"),
                    "position": c.get("position"),
                    "sub_position": c.get("sub_position"),
                    "type": c.get("type"),
                    "is_scanned": c.get("is_scanned"),
                    "text": c.get("text") or c.get("data") or "",
                    "embedding": emb.tolist() if hasattr(emb, "tolist") else emb,
                }
            )
        logger.debug("embed_chunks prepared %d embedding dicts", len(out))
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Failed packaging embeddings into output dicts: %s", e)
        raise

    return out
