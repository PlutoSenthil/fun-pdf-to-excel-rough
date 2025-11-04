import os
import json
import tempfile
from io import BytesIO
import pandas as pd
from google import genai
from google.genai import types

DEFAULT_PROMPT = (
    "Extract the main transaction table from this bank statement PDF. "
    "Return JSON with columns: date, description, withdrawal_amount, credit_amount, balance."
)

class GeminiExtractor:
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY_1")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def process_pdf(self, pdf_bytes: bytes, prompt: str = DEFAULT_PROMPT) -> BytesIO:
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            # Upload file to Gemini
            file_obj = self.client.files.upload(file=tmp_path)

            # Build request
            user_message = types.Content(
                role="user",
                parts=[prompt, file_obj]
            )

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[user_message],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            # Parse JSON
            data = json.loads(response.text)
            df = pd.DataFrame(data.get("transaction_data", data))  # fallback if schema not enforced

            # Convert to Excel
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Transactions")
            excel_buf.seek(0)
            return excel_buf

        finally:
            try:
                if 'file_obj' in locals():
                    self.client.files.delete(name=file_obj.name)
            except:
                pass
            os.remove(tmp_path)