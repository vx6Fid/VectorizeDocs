from fastapi.middleware.cors import CORSMiddleware
import os
import gc
import asyncio
import logging
import traceback
import pdfplumber
import requests
from io import BytesIO
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from utils.pdf_processing import process_pdf_batch
from utils.mongo_utils import vector_collection
from utils.s3_utils import list_s3_pdfs, fetch_pdf


PDF_BATCH_SIZE = 20
GPU_SERVER_URL = "http://127.0.0.1:9000/enqueue"

# Configure logging with timestamps and level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tender_processor")


async def process_single_tender(payload: dict[str, Any]) -> dict[str, Any]:
    tender_id = payload["tender_id"]
    logger.debug("Starting Tender Processing for ID: %s", tender_id)
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

        try:
            logger.debug("Checking for existing documents in Mongo for %s / %s", tender_id, document_name)
            existing = await asyncio.to_thread(
                vector_collection.find_one,
                {"tender_id": tender_id, "document_name": document_name},
                {"document_complete": 1}
            )
            logger.debug("Mongo count for %s: %s", document_name, exists)
            
            if existing and existing.get("document_complete"):
                logger.info("Skipping %s because it already exists in Mongo", document_name)
                result["skipped_docs"] += 1
                continue
            if existing:
                logger.info("Processing %s because it already exists in Mongo, but document incomplete", document_name)
                try:
                    await asyncio.to_thread(
                            vector_collection.delete_many,
                            {"tender_id": tender_id, "document_name": document_name}
                    ) 
                    logger.info("Removed partial embeddings from MongoDB...")
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.exception("Error removing existing embeddings from Mongo for %s: %s", document_name, e)
                    result["errors"].append(f"mongo_count_{document_name}: {str(e)}\n{tb}")
                    continue

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Error checking Mongo for %s: %s", document_name, e)
            result["errors"].append(f"mongo_count_{document_name}: {str(e)}\n{tb}")
            continue

        try:
            logger.debug("Fetching PDF from S3: %s", pdf_key)
            pdf_stream = await fetch_pdf(pdf_key)
            pdf_bytes = pdf_stream.read()
            logger.info("Fetched PDF %s (type=%s)", document_name, type(pdf_stream))
            total_pages = await asyncio.to_thread(
                lambda: len(pdfplumber.open(BytesIO(pdf_bytes)).pages)
            )
            logger.info(f"📄 Total pages: {total_pages}")
            if total_pages == 0:
                logger.info("⚠ Empty PDF, skipping")
                report["empty_docs"] += 1
                continue

            for start in range(0, total_pages, PDF_BATCH_SIZE):
                end = min(start + PDF_BATCH_SIZE, total_pages)
                is_last = (end >= total_pages)

                logger.debug(f"Page batch in thread: {start} → {end} (last={is_last})")

                # process_pdf contains blocking PDF parsing; run it in a separate thread/process
                # IMPORTANT: don't call asyncio.run from within an existing event loop / thread
                
                batch_result = await process_pdf_batch(pdf_bytes, start, end)

                if not isinstance(batch_result, dict):
                    raise TypeError(f"process_pdf returned non-dict: {type(pdf_result)}")

                chunks: List[Dict[str, Any]] = batch_result.get("chunks", [])
                scanned_pages = batch_result.get("scanned_pages", 0)
                regular_pages = batch_result.get("regular_pages", 0)
                result["scanned_pages"] += scanned_pages
                result["regular_pages"] += regular_pages
                logger.debug("batch_result contains %d chunks for %s", len(chunks), document_name)

                if chunks:
                        logger.info("   → Sending batch to GPU server...")
                        try:
                            resp = requests.post(GPU_SERVER_URL, json={
                                "chunks": chunks,
                                "document_name": document_name,
                                "tender_id": tender_id,
                                "is_last_batch": is_last
                            })
                            logger.info(f"     GPU Response: {resp.status_code}")
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.exception(f"❌ GPU enqueue failed: {e}")
                            result["errors"].append(f"{document_name}: {str(e)}\n{tb}")
                            continue
    
                try:
                    gc.collect()
                        logger.debug("gc.collect() called after processing")
                except Exception:
                    logger.exception("gc.collect() failed for batch")

            logger.info(f"✔ Completed queuing document: {document_name}")
            report["processed_docs"] += 1

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Unhandled exception processing %s: %s", document_name, e)
            result["errors"].append(f"{document_name}: {str(e)}\n{tb}")

    logger.info(f"\n🎯 Tender {tender_id} COMPLETED\n")
    return result
