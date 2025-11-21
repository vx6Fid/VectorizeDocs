import io
import gc
import re
import time
import base64
import requests
import pdfplumber
from PIL import Image
from groq import Groq
from io import BytesIO
from .config import GROQ_API_KEY, GROQ_OCR_PROMPT, DEEPSEEK_API_URL, DEEPSEEK_API_KEY, DEEPSEEK_TRANSLATE_PROMPT

groq_client = Groq(api_key=GROQ_API_KEY)

def clean_llm_output(text: str) -> str:
    text = re.sub(r"```(?:markdown)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", "", text, flags=re.DOTALL)
    return text.strip()

def render_page_to_image(page) -> bytes:
    image = page.to_image(resolution=200).original.convert("RGB")
    resized = image.resize((image.width // 2, image.height // 2))
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=40)
    return buffer.getvalue()

def query_deepseek(prompt, retries=3, delay=2):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a tender consultant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
            data = response.json()

            if "choices" in data:
                cleaned = clean_llm_output(data["choices"][0]["message"]["content"])
                return cleaned

            elif "error" in data:
                raise RuntimeError(data["error"]["message"])

            else:
                raise RuntimeError(f"Unexpected response: {data}")

        except Exception as e:
            print(f"DeepSeek error: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                raise

def query_groq(pil_image_bytes: bytes, prompt: str) -> str:
    img_base64 = base64.b64encode(pil_image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{img_base64}"

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        }],
        temperature=0.3,
        max_completion_tokens=4096
    )
    return response.choices[0].message.content.strip()

def is_scanned_page(page):
    text = page.extract_text() or ""
    return len(text.strip()) < 10

def process_scanned_page_worker(args):
    page_num, pdf_bytes = args
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[page_num]
            print(f"\n[SCANNED PAGE] Processing Page {page_num+1}")

            image_bytes = render_page_to_image(page)

            try:
                raw_content = query_groq(image_bytes, GROQ_OCR_PROMPT)
                print(f"\n📷 [SCANNED PAGE] Page {page_num+1}, raw content length: {len(raw_content)}")
            except Exception as e_groq:
                raw_content = f"<!-- Groq error: {e_groq} -->"

            if not isinstance(raw_content, str) or raw_content is None:
                raw_content = ""

            del image_bytes
            gc.collect()
            return {"page": page_num+1, "raw_content": raw_content}

    except Exception as e:
        return {"page": page_num+1, "raw_content": f"<!-- Error: {e} -->"}

def deepseek_translate_worker(args):
    page_num, raw_text = args
    try:
        prompt = f"{DEEPSEEK_TRANSLATE_PROMPT}\n\nText to translate:\n{raw_text}"
        translated_text = query_deepseek(prompt)
        return {"page": page_num, "translated_text": translated_text}
    except Exception as e:
        return {"page": page_num, "translated_text": f"<!-- DeepSeek error: {e} -->"}
