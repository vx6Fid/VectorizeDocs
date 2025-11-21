from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL_NAME, device, BATCH_SIZE

model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

def embed_batch(chunks):
    texts = [c["data"] for c in chunks]
    vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()

    out = []
    for c, emb in zip(chunks, vectors):
        out.append({
            "tender_id": c["tender_id"],
            "document_name": c["document_name"],
            "page": c["page"],
            "position": c["position"],
            "sub_position": c["sub_position"],
            "type": c["type"],
            "is_scanned": c["is_scanned"],
            "text": c["data"],
            "embedding": emb
        })

    return out
