# processor.py
import asyncio
import gc
import logging
import os
import traceback
from io import BytesIO
from typing import Any

import pdfplumber

from gpu_worker import embedding_queue
from utils.mongo_utils import vector_collection
from utils.pdf_processing import process_pdf_batch
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

        # === Mongo semantics: check for document_complete ===
        try:
            # Check if this document is already marked complete
            complete_doc = await asyncio.to_thread(
                lambda: vector_collection.find_one(
                    {
                        "tender_id": tender_id,
                        "document_name": document_name,
                        "document_complete": True,
                    }
                )
            )

            if complete_doc:
                logger.info("Skipping %s because document_complete=True", document_name)
                result["skipped_docs"] += 1
                continue

            # Remove any partial embeddings (if present)
            logger.info("Removing partial embeddings for %s (if any)", document_name)
            await asyncio.to_thread(
                vector_collection.delete_many,
                {"tender_id": tender_id, "document_name": document_name},
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception(
                "Error checking/cleaning Mongo for %s: %s", document_name, e
            )
            result["errors"].append(f"mongo_check_{document_name}: {str(e)}\n{tb}")
            # continue to next pdf rather than fail everything
            continue

        # === Fetch PDF bytes ===
        try:
            logger.debug("Fetching PDF from S3: %s", pdf_key)
            pdf_stream = await fetch_pdf(pdf_key)
            pdf_bytes = pdf_stream.read()
            logger.info("Fetched PDF %s size=%d bytes", document_name, len(pdf_bytes))
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Failed to fetch PDF %s: %s", document_name, e)
            result["errors"].append(f"fetch_{document_name}: {str(e)}\n{tb}")
            continue

        # === Determine total pages using pdfplumber ===
        try:
            total_pages = await asyncio.to_thread(
                lambda: len(pdfplumber.open(BytesIO(pdf_bytes)).pages)
            )
            logger.info("%s total_pages=%d", document_name, total_pages)
            if total_pages == 0:
                logger.warning(
                    "Empty PDF %s, marking empty and skipping", document_name
                )
                result["empty_docs"] += 1
                # mark empty PDF as complete so we don't reprocess forever
                try:
                    await asyncio.to_thread(
                        lambda: vector_collection.update_one(
                            {"tender_id": tender_id, "document_name": document_name},
                            {"$set": {"document_complete": True}},
                            upsert=True,
                        )
                    )
                except Exception as e2:
                    tb2 = traceback.format_exc()
                    logger.exception(
                        "Failed to mark empty PDF complete %s: %s", document_name, e2
                    )
                    result["errors"].append(
                        f"mark_empty_complete_{document_name}: {str(e2)}\n{tb2}"
                    )
                continue
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Failed to count pages for %s: %s", document_name, e)
            result["errors"].append(f"pagecount_{document_name}: {str(e)}\n{tb}")
            continue

        # === Dynamic batch size decision (same rule as embed_server.py) ===
        try:
            file_size_kb = len(pdf_bytes) / 1024
            size_per_page_kb = file_size_kb / max(total_pages, 1)
            batch_size = 20 if size_per_page_kb < 250 else 5
            logger.info(
                "%s dynamic batch_size=%d (size_per_page=%.1f KB)",
                document_name,
                batch_size,
                size_per_page_kb,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Failed computing batch size for %s: %s", document_name, e)
            result["errors"].append(f"batchsize_{document_name}: {str(e)}\n{tb}")
            # fallback
            batch_size = 5

        # === Process by page batches and enqueue to internal GPU worker ===
        try:
            for start in range(0, total_pages, batch_size):
                end = min(start + batch_size, total_pages)
                is_last = end >= total_pages

                logger.info(
                    "%s page batch %d -> %d (last=%s)",
                    document_name,
                    start,
                    end,
                    is_last,
                )

                # process_pdf_batch is ASYNC → call directly and expect (chunks, scanned, regular)
                chunks, scanned, regular = await process_pdf_batch(
                    pdf_bytes, start, end
                )

                logger.debug(
                    "   • Chunks = %d | Scanned = %d | Regular = %d",
                    len(chunks),
                    scanned,
                    regular,
                )

                result["scanned_pages"] += scanned
                result["regular_pages"] += regular

                if not chunks:
                    logger.warning(
                        "No chunks produced for %s batch %d-%d",
                        document_name,
                        start,
                        end,
                    )
                    if is_last:
                        # last batch produced no chunks → treat as empty; mark complete
                        result["empty_docs"] += 1
                        try:
                            await asyncio.to_thread(
                                lambda: vector_collection.update_one(
                                    {
                                        "tender_id": tender_id,
                                        "document_name": document_name,
                                    },
                                    {"$set": {"document_complete": True}},
                                    upsert=True,
                                )
                            )
                        except Exception as e2:
                            tb2 = traceback.format_exc()
                            logger.exception(
                                "Failed to mark empty final batch complete for %s: %s",
                                document_name,
                                e2,
                            )
                            result["errors"].append(
                                f"mark_empty_final_{document_name}: {str(e2)}\n{tb2}"
                            )
                    continue

                # ENQUEUE TO INTERNAL GPU WORKER (follows Phase 1 contract)
                try:
                    embedding_queue.put((chunks, document_name, tender_id, is_last))
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.exception(
                        "Failed to enqueue batch for %s: %s", document_name, e
                    )
                    result["errors"].append(f"enqueue_{document_name}: {str(e)}\n{tb}")
                    # continue processing remaining batches (do not abort the whole tender)
                    continue

                # small cleanup to release memory
                try:
                    del chunks
                    gc.collect()
                except Exception:
                    logger.exception("Cleanup failed for %s", document_name)

            # If we reached here, we queued all batches for the document
            logger.info("Completed queuing document: %s", document_name)
            result["processed_docs"] += 1

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
