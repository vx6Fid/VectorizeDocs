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
    pdf_keys = asyncio.run(list_s3_pdfs(s3_prefix))

    for pdf_key in pdf_keys:
        document_name = os.path.basename(pdf_key)

        # Skip existing
        exists = vector_collection.count_documents(
            {"tender_id": tender_id, "document_name": document_name}
        )
        if exists > 0:
            result["skipped_docs"] += 1
            continue

        try:
            pdf_stream = asyncio.run(fetch_pdf(pdf_key))
            pdf_result = await process_pdf(pdf_stream)
            chunks = pdf_result["chunks"]

            result["scanned_pages"] += pdf_result["scanned_pages"]
            result["regular_pages"] += pdf_result["regular_pages"]

            if not chunks:
                result["empty_docs"] += 1
                continue

            # *** NO QUEUE — direct GPU embedding ***
            embeddings = embed_chunks(chunks)

            # Insert into Mongo
            docs = []
            for chunk, emb in zip(chunks, embeddings):
                docs.append(
                    {
                        "tender_id": tender_id,
                        "document_name": document_name,
                        "text": chunk["text"],
                        "embedding": emb,
                    }
                )

            if docs:
                vector_collection.insert_many(docs)
                result["processed_docs"] += 1

        except Exception as e:
            result["errors"].append(f"{document_name}: {str(e)}")

        gc.collect()

    return result
