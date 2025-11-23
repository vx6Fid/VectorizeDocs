import gc
import threading
from queue import Queue

from utils.embedding_utils import embed_batch
from utils.mongo_utils import store_embeddings_in_db, vector_collection

# Internal embedding queue (shared with processor)
embedding_queue = Queue(maxsize=20000)

# Signal for clean shutdown
STOP_SIGNAL = object()


def gpu_worker():
    print("[Info] Internal GPU worker started")

    while True:
        task = embedding_queue.get()

        if task is STOP_SIGNAL:
            print("[Info] GPU worker stopping")
            break

        chunks, document_name, tender_id, is_last_batch = task

        try:
            # Add metadata to each chunk (same as original GPU server)
            for c in chunks:
                c["tender_id"] = tender_id
                c["document_name"] = document_name

            # Perform embedding
            embeddings = embed_batch(chunks)

            # Store embedding vectors in Mongo
            store_embeddings_in_db(embeddings, document_name, tender_id)

            # Mark document complete if this is the final batch
            if is_last_batch:
                vector_collection.update_one(
                    {"tender_id": tender_id, "document_name": document_name},
                    {"$set": {"document_complete": True}},
                    upsert=True,
                )

        except Exception as e:
            print(f"[GPU WORKER] Error processing {document_name}: {e}")

        gc.collect()
        embedding_queue.task_done()


def start_gpu_worker():
    """
    Spawns the GPU worker thread in daemon mode.
    Must be called once at program startup (e.g. in python_worker.py).
    """
    thread = threading.Thread(target=gpu_worker, daemon=True)
    thread.start()
    print("[Info] GPU worker thread started")
    return thread


def stop_gpu_worker():
    embedding_queue.put(STOP_SIGNAL)


def join_gpu_worker(thread):
    thread.join()
