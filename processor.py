import asyncio
import gc
import os

from gpu_embedder import embed_chunks
from utils.mongo_utils import vector_collection
from utils.pdf_processing import process_pdf
from utils.s3_utils import fetch_pdf, list_s3_pdfs


async def process_single_tender(payload):
    tender_id = payload["tender_id"]
    result = {
        "tender_id": tender_id,
        "processed_docs": 0,
        "skipped_docs": 0,
        "empty_docs": 0,
        "scanned_pages": 0,
        "regular_pages": 0,
        "errors": [],
    }

    s3_prefix = f"tender-documents/{tender_id}/"
    pdf_keys = await list_s3_pdfs(s3_prefix)

    for pdf_key in pdf_keys:
        document_name = os.path.basename(pdf_key)

        # Count documents in Mongo in a thread to avoid blocking the event loop
        exists = await asyncio.to_thread(
            lambda: vector_collection.count_documents(
                {"tender_id": tender_id, "document_name": document_name}
            )
        )
        if exists > 0:
            result["skipped_docs"] += 1
            continue

        try:
            pdf_stream = await fetch_pdf(pdf_key)

            # process_pdf contains blocking PDF parsing; run it in a separate thread/process so the event loop is not blocked
            pdf_result = await asyncio.to_thread(
                lambda: asyncio.run(process_pdf(pdf_stream))
            )

            chunks = pdf_result["chunks"]

            result["scanned_pages"] += pdf_result.get("scanned_pages", 0)
            result["regular_pages"] += pdf_result.get("regular_pages", 0)

            if not chunks:
                result["empty_docs"] += 1
                continue

            # Run embedding in a thread so heavy CPU/GPU work does not block the loop.
            # Ideally this is a separate GPU process; as a stop-gap we offload to a thread.
            embeddings = await asyncio.to_thread(embed_chunks, chunks)

            # Insert into Mongo in a thread
            docs = []
            for chunk, emb in zip(chunks, embeddings):
                docs.append(
                    {
                        "tender_id": tender_id,
                        "document_name": document_name,
                        "text": chunk["text"],
                        "embedding": emb.tolist() if hasattr(emb, "tolist") else emb,
                    }
                )

            if docs:
                await asyncio.to_thread(vector_collection.insert_many, docs)
                result["processed_docs"] += 1

        except Exception as e:
            result["errors"].append(f"{document_name}: {str(e)}")

        # free memory regularly
        gc.collect()

    return result
