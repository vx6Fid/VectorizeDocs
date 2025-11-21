import boto3
from io import BytesIO
from .config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET
import asyncio

# Synchronous boto3 client
_s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

async def list_s3_pdfs(prefix: str):
    def _list():
        paginator = _s3_client.get_paginator("list_objects_v2")
        pdf_keys = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(".pdf"):
                    pdf_keys.append(obj["Key"])
        return pdf_keys

    return await asyncio.to_thread(_list)

async def fetch_pdf(key: str) -> BytesIO:
    def _fetch():
        obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return BytesIO(obj["Body"].read())

    return await asyncio.to_thread(_fetch)
