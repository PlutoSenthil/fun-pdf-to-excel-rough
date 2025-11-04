# gemini.py
import os
import json
import time
import tempfile
from io import BytesIO
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from google import genai
from google.genai import types


# ---- Pydantic Models (same fields you defined) ----

class BankTransactionRow(BaseModel):
    date: str = Field(description="The date of the transaction (e.g., '30/10/2025').")
    reference_or_cheque_no: Optional[str] = Field(
        None, description="The unique transaction reference number, cheque number, or UPI ID/Ref. Null if not provided."
    )
    description: str = Field(description="The full narration or transaction details.")
    withdrawal_amount: Optional[float] = Field(
        None, description="The amount withdrawn/debited. Null if it's a credit. Must be a positive number."
    )
    credit_amount: Optional[float] = Field(
        None, description="The amount deposited/credited. Null if it's a debit. Must be a positive number."
    )
    balance: float = Field(description="The running balance of the account *after* this transaction is processed.")


class ExtractedBankStatement(BaseModel):
    # HEADER INFORMATION
    bank_name: str = Field(description="The full name of the bank.")
    branch_name: str = Field(description="The name of the branch (e.g., 'Br' or 'Branch' detail).")
    account_holder_name: str = Field(description="The full name of the primary account holder.")
    statement_period: str = Field(description="The period covered by the statement (e.g., '01/10/2025 to 31/10/2025').")
    initial_balance: float = Field(description="The opening/initial balance of the account at the start of the statement period (often labeled 'Init. Bal.').")

    # TRANSACTION DATA
    transaction_data: List[BankTransactionRow]

    # FOOTER/SUMMARY INFORMATION
    total_debit_amount: float = Field(description="The grand total of all withdrawal/debit amounts in the statement ('TRANSACTION TOTAL of debit').")
    total_credit_amount: float = Field(description="The grand total of all deposit/credit amounts in the statement ('TRANSACTION TOTAL of credit').")
    closing_balance: float = Field(description="The final balance of the account at the end of the statement period.")


DEFAULT_PROMPT = (
    "You are an expert data extraction bot. From the uploaded bank statement PDF, "
    "extract structured data strictly following the provided JSON schema. "
    "Rules:\n"
    "- Follow field names and types exactly.\n"
    "- Use numbers for amounts (no currency symbols), positive values only.\n"
    "- Keep dates as they appear in the statement.\n"
    "- Use null for missing values.\n"
    "- Do not invent or hallucinate data not present in the PDF.\n"
)


class GeminiExtractor:
    """
    Handles uploading PDFs to Google Gemini File API, requesting a structured JSON
    that conforms to the Pydantic schema, and converting it to Excel.
    """

    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash", prompt: str = DEFAULT_PROMPT):
        if not api_key:
            raise ValueError("A valid GOOGLE_API_KEY is required.")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.prompt = prompt

    # ---------- Internal helpers ----------

    def _upload_pdf(self, pdf_path: str):
        """Upload a local PDF path to Gemini's File API."""
        return self.client.files.upload(file=pdf_path)

    def _call_model_for_json(self, file_obj) -> str:
        """
        Ask Gemini to return structured JSON strictly per ExtractedBankStatement schema.
        Returns a JSON string.
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[self.prompt, file_obj],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedBankStatement,
            ),
        )
        return response.text

    def _json_to_dataframe(self, extracted: dict) -> pd.DataFrame:
        txns = extracted.get("transaction_data", []) or []
        df = pd.DataFrame(txns)
        # Clean up numeric fields if needed
        numeric_cols = ["withdrawal_amount", "credit_amount", "balance"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _dataframe_to_excel_bytes(self, df: pd.DataFrame, metadata: Optional[dict] = None) -> BytesIO:
        """
        Convert DataFrame (+ optional metadata) to an Excel workbook in memory.
        Sheet1: Transactions
        Sheet2: Header/Footer summary
        """
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Transactions
            (df or pd.DataFrame()).to_excel(writer, index=False, sheet_name="Transactions")

            # Summary sheet (optional but handy)
            if metadata:
                meta_pairs = []
                # Safely collect a few known fields
                for key in [
                    "bank_name", "branch_name", "account_holder_name",
                    "statement_period", "initial_balance",
                    "total_debit_amount", "total_credit_amount", "closing_balance"
                ]:
                    if key in metadata:
                        meta_pairs.append({"Field": key, "Value": metadata[key]})
                if meta_pairs:
                    pd.DataFrame(meta_pairs).to_excel(writer, index=False, sheet_name="Statement_Summary")

        buf.seek(0)
        return buf

    # ---------- Public API ----------

    def process_pdf_to_excel(self, pdf_bytes: bytes, original_filename: str) -> Tuple[dict, BytesIO]:
        """
        Accept in-memory PDF bytes, upload via temp file, get structured JSON, validate
        with Pydantic, convert to DataFrame, and return the Excel bytes.
        """
        # Create a temp file because the File API expects a path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            file_obj = self._upload_pdf(tmp_path)

            # Retry strategy for occasional transient errors
            max_attempts = 3
            backoff = 2
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    raw_json = self._call_model_for_json(file_obj)
                    data = json.loads(raw_json)

                    # Validate against Pydantic schema to ensure consistency
                    validated = ExtractedBankStatement(**data).model_dump()
                    df = self._json_to_dataframe(validated)
                    excel_io = self._dataframe_to_excel_bytes(df, metadata=validated)
                    return validated, excel_io
                except (json.JSONDecodeError, ValidationError) as e:
                    last_err = e
                    if attempt < max_attempts:
                        time.sleep(backoff ** attempt)
                    else:
                        raise
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts:
                        time.sleep(backoff ** attempt)
                    else:
                        raise
        finally:
            # Best-effort cleanup of Gemini temp file and local temp
            try:
                if 'file_obj' in locals() and getattr(file_obj, "name", None):
                    self.client.files.delete(name=file_obj.name)
            except Exception:
                pass
            try:
                os.remove(tmp_path)
            except Exception:
                pass