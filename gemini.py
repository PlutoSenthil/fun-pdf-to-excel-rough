import io
import json
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ----------------- Pydantic Models -----------------

class FinancialTransactionRow(BaseModel):
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


class ExtractedFinancialStatement(BaseModel):
    # HEADER INFORMATION
    institution_name: str = Field(description="The full name of the financial institution.")
    branch_name: str = Field(description="The name of the branch (e.g., 'Br' or 'Branch' detail).")
    account_holder_name: str = Field(description="The full name of the primary account holder.")
    statement_period: str = Field(description="The period covered by the statement (e.g., '01/10/2025 to 31/10/2025').")
    initial_balance: float = Field(description="The opening/initial balance at the start of the statement period.")

    # TRANSACTION DATA
    transaction_data: List[FinancialTransactionRow]

    # FOOTER/SUMMARY INFORMATION
    total_debit_amount: float = Field(
        description="The grand total of all withdrawal/debit amounts in the statement ('TRANSACTION TOTAL of debit')."
    )
    total_credit_amount: float = Field(
        description="The grand total of all deposit/credit amounts in the statement ('TRANSACTION TOTAL of credit')."
    )
    closing_balance: float = Field(description="The final balance of the account at the end of the statement period.")

    # OPTIONAL COUNTS
    debit_count: Optional[int] = Field(
        None, description="The total number of debit transactions in the statement."
    )
    credit_count: Optional[int] = Field(
        None, description="The total number of credit transactions in the statement."
    )


# ----------------- Core extraction -----------------

PROMPT = (
    "You are validating bank statement table rows and returning a structured JSON that matches the provided schema.\n"
    "Rules:\n"
    "1) Do not fabricate values.\n"
    "2) Keep date formats as-is.\n"
    "3) Ensure correct withdrawal_amount and credit_amount per row (exactly one is non-null when applicable).\n"
    "4) Use numeric types for amounts and balance.\n"
    "5) The running balance is the balance after each transaction.\n"
    "6) Include total_debit_amount, total_credit_amount, closing_balance, and if available, debit_count and credit_count.\n"
    "7) Return JSON only (no markdown, no code fences, no commentary).\n"
    "8) If a field is not present in the PDF, use empty string for text fields and null for numeric fields.\n"
    "9) Accuracy is critical; do not infer values that aren't present.\n"
)

def extract_financial_statement(
    api_key: str,
    model_id: str,
    pdf_file_path: str,
    file_bytes: bytes,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Calls Gemini and returns:
      - extracted_data (dict) on success,
      - raw_text (str) of model output (for download if parsing fails),
      - error (str) on failure.
    """
    try:
        if not api_key:
            return None, None, "Missing API key."

        client = genai.Client(api_key=api_key)

        content_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf",
        )

        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, content_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFinancialStatement,
            ),
        )

        # Prefer structured parse if SDK provides it
        if hasattr(response, "parsed") and response.parsed:
            parsed = response.parsed
            if isinstance(parsed, dict):
                return parsed, None, None
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump(), None, None

        # Else try text JSON
        raw_text = getattr(response, "text", None)
        if raw_text:
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    return data, raw_text, None
            except Exception:
                # parsing failed; return raw_text for download
                return None, raw_text, "Could not parse JSON response."
        else:
            # Try candidates (raw text) for download
            raw_text_dump = []
            try:
                for c in getattr(response, "candidates", []) or []:
                    for p in getattr(getattr(c, "content", None), "parts", []) or []:
                        t = getattr(p, "text", None)
                        if t:
                            raw_text_dump.append(t)
            except Exception:
                pass
            raw_text_joined = "\n\n".join(raw_text_dump) if raw_text_dump else None
            return None, raw_text_joined, "Empty response."

    except Exception:
        return None, None, "Extraction error."

    return None, None, "Unknown error."


# ----------------- Excel export -----------------

def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    """
    Writes:
      - Transactions sheet
      - Summary sheet (all non-transaction fields)
    """
    df = pd.DataFrame(json_data.get("transaction_data", []))
    meta = {k: v for k, v in json_data.items() if k != "transaction_data"}

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Transactions", index=False)
        meta_df = pd.DataFrame(list(meta.items()), columns=["Field", "Value"])
        meta_df.to_excel(writer, sheet_name="Summary", index=False)

        # Auto widths
        ws_t = writer.sheets["Transactions"]
        for i, col in enumerate(df.columns, start=1):
            width = max(len(col), int(df[col].astype(str).str.len().max() if not df.empty else 0)) + 2
            ws_t.column_dimensions[ws_t.cell(row=1, column=i).column_letter].width = width

        ws_s = writer.sheets["Summary"]
        for i, col in enumerate(meta_df.columns, start=1):
            width = max(len(col), int(meta_df[col].astype(str).str.len().max() if not meta_df.empty else 0)) + 2
            ws_s.column_dimensions[ws_s.cell(row=1, column=i).column_letter].width = width

    output.seek(0)
    return output