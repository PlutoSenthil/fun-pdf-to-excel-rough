import io
import json
import re
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- Schema ---
class FinancialTransactionRow(BaseModel):
    date: str
    reference_or_cheque_no: Optional[str] = None
    description: str
    withdrawal_amount: Optional[float] = None
    credit_amount: Optional[float] = None
    balance: float

class ExtractedFinancialStatement(BaseModel):
    institution_name: str
    branch_name: str
    account_holder_name: str
    statement_period: str
    initial_balance: float
    transaction_data: List[FinancialTransactionRow]
    total_debit_amount: float
    total_credit_amount: float
    closing_balance: float
    debit_count: Optional[int] = None
    credit_count: Optional[int] = None

# --- JSON salvage ---
def _clean_json_text(s: str) -> str:
    s = s.replace("\ufeff", "")
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.MULTILINE)
    return s

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # Try to find first balanced braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end+1])
            except Exception:
                return None
    return None

# --- Core extraction ---
def extract_financial_statement(api_key: str, model_id: str, file_bytes: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        client = genai.Client(api_key=api_key)
        pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")

        PROMPT = (
            "Extract all header info, transaction rows, and summary from this bank statement PDF.\n"
            "Rules:\n"
            "- Do not fabricate values.\n"
            "- Keep date formats as-is.\n"
            "- Ensure correct withdrawal_amount or credit_amount per row.\n"
            "- Use numeric types for amounts and balance.\n"
            "- Return JSON only matching schema."
        )

        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, pdf_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFinancialStatement,
            ),
        )

        # Parse response
        if hasattr(response, "parsed") and response.parsed:
            return response.parsed.model_dump(), None
        if hasattr(response, "text") and response.text:
            data = _extract_json(response.text)
            return (data, None) if data else (None, "Could not parse JSON.")
        return None, "Empty response."
    except Exception as e:
        return None, f"Error: {e}"

# --- Excel export ---
def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    df = pd.DataFrame(json_data.get("transaction_data", []))
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Transactions", index=False)
        summary = {k: v for k, v in json_data.items() if k != "transaction_data"}
        pd.DataFrame(list(summary.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Summary", index=False)
    output.seek(0)
    return output
