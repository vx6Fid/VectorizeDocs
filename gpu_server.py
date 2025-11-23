from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from queue import Queue
import threading
import gc
from utils.embedding_utils import embed_batch
from utils.mongo_utils import store_embeddings_in_db, vector_collection
import torch

embedding_queue = Queue(maxsize=20000)
STOP_SIGNAL = object()

app = FastAPI(title="GPU Embedding Server")

def gpu_worker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 GPU worker started on: {device}")

    while True:
        task = embedding_queue.get()
        if task is STOP_SIGNAL:
            print("🛑 GPU worker stopping (STOP signal)")
            break

        chunks, document_name, tender_id, is_last_batch = task
        for c in chunks:
            c["tender_id"] = tender_id
            c["document_name"] = document_name

        try:
            embeddings = embed_batch(chunks)
            print(f"[{document_name}] 🔹 Vectorized {len(chunks)} chunks")

            store_embeddings_in_db(embeddings, document_name, tender_id)
            print(f"[{document_name}] 💾 Stored in MongoDB")

            if is_last_batch:
                vector_collection.update_one(
                    {"tender_id": tender_id, "document_name": document_name},
                    {"$set": {"document_complete": True}},
                    upsert=True
                )
                print(f"[{document_name}] 🎉 Document marked COMPLETE")

        except Exception as e:
            print(f"[GPU WORKER] ❌ Error: {document_name}: {e}")

        gc.collect()
        embedding_queue.task_done()

@app.on_event("startup")
def start_worker():
    thread = threading.Thread(target=gpu_worker, daemon=True)
    thread.start()
    print("✅ GPU worker thread started")

@app.on_event("shutdown")
def stop_worker():
    embedding_queue.put(STOP_SIGNAL)
    print("🛑 Stop signal sent to GPU worker")

class EmbedRequest(BaseModel):
    chunks: list
    document_name: str
    tender_id: str
    is_last_batch: bool

@app.post("/enqueue")
def enqueue_embedding(req: EmbedRequest):
    try:
        embedding_queue.put(
            (req.chunks, req.document_name, req.tender_id, req.is_last_batch)
        )
        return {"status": "queued", "document_name": req.document_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
