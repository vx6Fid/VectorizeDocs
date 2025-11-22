import asyncio
import gc
import logging
import os
import traceback
from typing import Any, Dict, List

from gpu_embedder import embed_chunks
from utils.mongo_utils import vector_collection
from utils.pdf_processing import process_pdf
from utils.s3_utils import fetch_pdf, list_s3_pdfs

# Configure logging with timestamps and level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tender_processor")


async def process_single_tender(payload: dict[str, Any]) -> dict[str, Any]:
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
    try:
        logger.debug("Listing S3 PDFs with prefix: %s", s3_prefix)
        pdf_keys = await list_s3_pdfs(s3_prefix)
        logger.info("Found %d pdf keys for tender %s", len(pdf_keys), tender_id)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Failed to list S3 PDFs for prefix %s: %s", s3_prefix, e)
        result["errors"].append(f"list_s3_pdfs: {str(e)}\n{tb}")
        return result

    for pdf_key in pdf_keys:
        document_name = os.path.basename(pdf_key)
        logger.debug("Processing pdf_key=%s document_name=%s", pdf_key, document_name)

        # Count documents in Mongo in a thread to avoid blocking the event loop
        try:
            logger.debug(
                "Checking for existing documents in Mongo for %s / %s",
                tender_id,
                document_name,
            )
            exists = await asyncio.to_thread(
                lambda: vector_collection.count_documents(
                    {"tender_id": tender_id, "document_name": document_name}
                )
            )
            logger.debug("Mongo count for %s: %s", document_name, exists)
            if exists > 0:
                result["skipped_docs"] += 1
                logger.info(
                    "Skipping %s because it already exists in Mongo", document_name
                )
                continue
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Error checking Mongo for %s: %s", document_name, e)
            result["errors"].append(f"mongo_count_{document_name}: {str(e)}\n{tb}")
            # continue to next pdf rather than fail everything
            continue

        try:
            logger.debug("Fetching PDF from S3: %s", pdf_key)
            pdf_stream = await fetch_pdf(pdf_key)
            logger.info("Fetched PDF %s (type=%s)", document_name, type(pdf_stream))

            # process_pdf contains blocking PDF parsing; run it in a separate thread/process
            # IMPORTANT: don't call asyncio.run from within an existing event loop / thread
            logger.debug("Calling process_pdf in a thread for %s", document_name)
            pdf_result = await asyncio.to_thread(process_pdf, pdf_stream)
            logger.info("process_pdf finished for %s", document_name)

            # Validate pdf_result structure
            if not isinstance(pdf_result, dict):
                raise TypeError(f"process_pdf returned non-dict: {type(pdf_result)}")

            chunks: List[Dict[str, Any]] = pdf_result.get("chunks", [])
            logger.debug(
                "pdf_result contains %d chunks for %s", len(chunks), document_name
            )

            scanned_pages = pdf_result.get("scanned_pages", 0)
            regular_pages = pdf_result.get("regular_pages", 0)
            result["scanned_pages"] += scanned_pages
            result["regular_pages"] += regular_pages
            logger.info(
                "Pages for %s -> scanned: %s, regular: %s",
                document_name,
                scanned_pages,
                regular_pages,
            )

            if not chunks:
                result["empty_docs"] += 1
                logger.warning("No chunks extracted for %s", document_name)
                continue

            # Print a sample of the first chunk keys / text length to debug schema mismatch
            try:
                first = chunks[0]
                logger.debug("First chunk keys: %s", list(first.keys()))
                sample_text = (
                    first.get("text") or first.get("data") or "<no-text-field>"
                )
                logger.debug(
                    "First chunk sample length: %d",
                    len(sample_text) if isinstance(sample_text, str) else -1,
                )
            except Exception:
                logger.exception("Failed to inspect first chunk for %s", document_name)

            # Run embedding in a thread so heavy CPU/GPU work does not block the loop.
            # If you have a GPU embedder (external) use it; otherwise use local embed_chunks below.
            try:
                logger.debug(
                    "Starting embedding for %s with %d chunks",
                    document_name,
                    len(chunks),
                )
                # If you have an external GPU embedder function, use it; otherwise fallback to local embed_chunks
                # We call it in a thread to avoid blocking
                embeddings = await asyncio.to_thread(embed_chunks, chunks)
                logger.info(
                    "Embedding finished for %s: received %d embeddings",
                    document_name,
                    len(embeddings),
                )
            except Exception as e:
                tb = traceback.format_exc()
                logger.exception("Embedding failed for %s: %s", document_name, e)
                result["errors"].append(f"embed_{document_name}: {str(e)}\n{tb}")
                continue

            # Build docs for Mongo insert
            docs = []
            try:
                for c, emb in zip(chunks, embeddings):
                    # If embedding returns dicts (like our embed_chunks below), handle both shapes
                    if isinstance(emb, dict) and "embedding" in emb:
                        embedding_vector = emb["embedding"]
                    else:
                        embedding_vector = emb

                    doc = {
                        "tender_id": tender_id,
                        "document_name": document_name,
                        "text": c.get("text") or c.get("data") or "",
                        "embedding": embedding_vector.tolist()
                        if hasattr(embedding_vector, "tolist")
                        else embedding_vector,
                        # preserve chunk metadata if present:
                        "chunk_meta": {
                            "page": c.get("page"),
                            "position": c.get("position"),
                            "sub_position": c.get("sub_position"),
                            "type": c.get("type"),
                            "is_scanned": c.get("is_scanned"),
                        },
                    }
                    docs.append(doc)
                logger.debug(
                    "Prepared %d docs for Mongo insert for %s", len(docs), document_name
                )
            except Exception as e:
                tb = traceback.format_exc()
                logger.exception(
                    "Failed to prepare docs for insert for %s: %s", document_name, e
                )
                result["errors"].append(f"prepare_docs_{document_name}: {str(e)}\n{tb}")
                continue

            # Insert into Mongo in a thread
            try:
                if docs:
                    logger.debug(
                        "Inserting %d docs into Mongo for %s", len(docs), document_name
                    )
                    await asyncio.to_thread(vector_collection.insert_many, docs)
                    result["processed_docs"] += 1
                    logger.info("Inserted docs into Mongo for %s", document_name)
                else:
                    logger.warning("No docs to insert for %s", document_name)
            except Exception as e:
                tb = traceback.format_exc()
                logger.exception("Mongo insert failed for %s: %s", document_name, e)
                result["errors"].append(f"mongo_insert_{document_name}: {str(e)}\n{tb}")
                continue

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Unhandled exception processing %s: %s", document_name, e)
            result["errors"].append(f"{document_name}: {str(e)}\n{tb}")

        finally:
            # free memory regularly
            try:
                gc.collect()
                logger.debug("gc.collect() called after processing %s", document_name)
            except Exception:
                logger.exception("gc.collect() failed for %s", document_name)

    return result
