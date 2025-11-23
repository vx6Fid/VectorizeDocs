import asyncio
import gc
from io import BytesIO

import pdfplumber

from utils.chunking import split_text_to_subchunks
from utils.config import MAX_PROCESSES_DEEPSEEK, MAX_PROCESSES_GROQ
from utils.regular_helpers import elements_to_positions, extract_page_content
from utils.scanned_helpers import (
    deepseek_translate_worker,
    is_scanned_page,
    process_scanned_page_worker,
)


async def groq_worker(job, semaphore):
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, process_scanned_page_worker, job)


async def deepseek_worker(job, semaphore):
    async with semaphore:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, deepseek_translate_worker, job)
        sub_chunks = split_text_to_subchunks(
            res["translated_text"], res["page"], 1, "text", is_scanned=True
        )
        gc.collect()
        return sub_chunks


async def process_pdf_batch(pdf_bytes, start_page=0, end_page=None):
    all_sub_chunks = []
    scanned_jobs = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        if end_page is None or end_page > total_pages:
            end_page = total_pages

        for i in range(start_page, end_page):
            page = pdf.pages[i]
            scanned = is_scanned_page(page)
            if scanned:
                scanned_jobs.append((i, pdf_bytes))
            else:
                elements = extract_page_content(page)
                positions = elements_to_positions(elements)
                for pos in positions:
                    sub_chunks = split_text_to_subchunks(
                        pos["content"],
                        i + 1,
                        pos["position"],
                        pos["type"],
                        is_scanned=False,
                    )
                    all_sub_chunks.extend(sub_chunks)
        gc.collect()

    groq_results = []
    if scanned_jobs:
        groq_semaphore = asyncio.Semaphore(MAX_PROCESSES_GROQ)
        groq_tasks = [groq_worker(job, groq_semaphore) for job in scanned_jobs]
        groq_results = await asyncio.gather(*groq_tasks)

    deepseek_results = []
    if groq_results:
        deepseek_jobs = [(res["page"], res["raw_content"]) for res in groq_results]
        deepseek_semaphore = asyncio.Semaphore(MAX_PROCESSES_DEEPSEEK)
        deepseek_tasks = [
            deepseek_worker(job, deepseek_semaphore) for job in deepseek_jobs
        ]
        deepseek_results = await asyncio.gather(*deepseek_tasks)
        for sub_chunks in deepseek_results:
            all_sub_chunks.extend(sub_chunks)
        gc.collect()

    gc.collect()
    return (
        all_sub_chunks,
        len(scanned_jobs),
        (end_page - start_page - len(scanned_jobs)),
    )
