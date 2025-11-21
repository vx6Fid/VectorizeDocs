import os

import torch
from dotenv import load_dotenv

load_dotenv()


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = os.getenv("DB_TENDER_BHARAT") or "TenderBharat"
TENDERS_COLLECTION = os.getenv("TENDERS_COLLECTION_NAME") or "Tenders"
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION_NAME") or "TenderDocs"

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 512

MAX_PROCESSES_GROQ = 5
MAX_PROCESSES_DEEPSEEK = 10

GROQ_OCR_PROMPT = """
Extract all text from this scanned page exactly as it appears on the page.
- Do NOT summarize, interpret, or add any commentary.
- Output only the text exactly as on the page, no less no more.
- If no text is found, return an empty string "".
"""

DEEPSEEK_TRANSLATE_PROMPT = """
You are a translator. Translate the following text to English exactly.
- If the text is already in English, leave it unchanged.
- Do not summarize, comment, or alter the content in any way.
- Preserve all formatting, spacing, and newlines.
- If the text is blank, return blank.
"""

RABBIT_URL = os.getenv("RABBITMQ_URL")
