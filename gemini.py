import os
import json
import tempfile
from io import BytesIO

import pandas as pd
from google import genai
from google.genai import types


DEFAULT_PROMPT = (
    "You are an expert data extraction bot. From this bank statement PDF, "
    "extract the main transactions table and return JSON with a list named "
    "'transaction_data'. Each item should have keys: "
    "date, description, withdrawal_amount, credit_amount, balance. "
    "Use numbers only for amounts (no currency symbol). Use null for missing."
)


class GeminiExtractor:
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY_1.")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def process_pdf_to_excel(self, pdf_bytes: bytes, prompt: str = DEFAULT_PROMPT) -> BytesIO:
        # 1) Save to a temp file (Files API wants a file path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        file_obj = None
        try:
            # 2) Upload to Gemini Files API
            file_obj = self.client.files.upload(file=tmp_path)

            # 3) Ask for structured JSON (no manual Content/Part objects here)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, file_obj],  # <- simple & valid
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            # 4) Parse JSON safely
            data = json.loads(response.text)

            # Accept either {"transaction_data":[...]} or a raw list [...]
            if isinstance(data, dict) and "transaction_data" in data:
                rows = data["transaction_data"]
            else:
                rows = data

            # 5) Create DataFrame
            df = pd.DataFrame(rows or [])

            # 6) Excel in memory
            out = BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Transactions")
            out.seek(0)
            return out

        finally:
            # best-effort cleanup
            try:
                if file_obj and getattr(file_obj, "name", None):
                    self.client.files.delete(name=file_obj.name)
            except Exception:
                pass
            try:
                os.remove(tmp_path)
            except Exception:
                pass