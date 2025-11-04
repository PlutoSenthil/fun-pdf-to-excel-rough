import io
import json
import re
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# --- Pydantic Models ---
# 1. Transaction Row Model (6 Columns)
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


# 2. Overall Statement Model (Includes Header/Footer Info)
class ExtractedFinancialStatement(BaseModel):
    # HEADER INFORMATION
    institution_name: str = Field(description="The full name of the financial institution.")
    branch_name: str = Field(description="The name of the branch (e.g., 'Br' or 'Branch' detail).")
    account_holder_name: str = Field(description="The full name of the primary account holder.")
    statement_period: str = Field(description="The period covered by the statement (e.g., '01/10/2025 to 31/10/2025').")
    initial_balance: float = Field(description="The opening/initial balance of the account at the start of the statement period (often labeled 'Init. Bal.').")

    # TRANSACTION DATA
    transaction_data: List[FinancialTransactionRow]

    # FOOTER/SUMMARY INFORMATION
    total_debit_amount: float = Field(description="The grand total of all withdrawal/debit amounts in the statement ('TRANSACTION TOTAL of debit').")
    total_credit_amount: float = Field(description="The grand total of all deposit/credit amounts in the statement ('TRANSACTION TOTAL of credit').")
    closing_balance: float = Field(description="The final balance of the account at the end of the statement period.")


# ----------------- JSON parsing hardener -----------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _strip_code_fences(s: str) -> str:
    # Remove markdown code fences like ```json ... ```
    return _CODE_FENCE_RE.sub("", s).strip()

def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first well-formed top-level JSON object from arbitrary text.
    Strategy:
      - Strip code fences
      - Scan for '{' and attempt to decode using a moving window that ends
        where braces balance to zero.
    """
    cleaned = _strip_code_fences(text)

    # Quick attempt: whole text might already be JSON
    whole = _try_json_loads(cleaned)
    if whole is not None:
        return whole

    # Scan for the largest valid JSON object
    start_positions = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in start_positions:
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : idx + 1]
                        obj = _try_json_loads(candidate)
                        if obj is not None:
                            return obj
        # If we exit loop, that start didn't terminate cleanly; try next start
    return None


# --- Core Function to Interact with Gemini ---

def extract_financial_statement(
    api_key: str,
    model_id: str,
    pdf_file_path: str,  # File path/handle from Streamlit (for display only)
    file_bytes: bytes,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts structured financial statement data from a PDF using the Gemini API
    via Part.from_bytes() (inline bytes). Best for files under ~20MB.

    Returns:
        (extracted_dict, None) on success; (None, error_message) on failure.
    """
    try:
        if not api_key:
            return None, "Missing API key."

        client = genai.Client(api_key=api_key)

        # Important: explicit mime_type; do NOT pass filename (not supported on some versions).
        pdf_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf",
        )

        PROMPT = (
            "You are an expert financial data extraction bot. "
            "From the uploaded Financial Statement PDF document, meticulously extract "
            "ALL header information (Institution name, holder name, period, initial balance), "
            "footer summaries (total debit/credit, closing balance), and the COMPLETE list "
            "of transactions from the main table, even if it spans multiple pages. "
            "Adhere strictly to the requested JSON schema. All amounts must be positive floats."
        )

        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, pdf_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFinancialStatement,
            ),
        )

        # ---------- Robust parsing ----------
        # 1) Prefer structured parse if present
        if hasattr(response, "parsed") and response.parsed:
            parsed = response.parsed
            if isinstance(parsed, dict):
                return parsed, None
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump(), None

        # 2) Try response.text (may contain extra formatting)
        if hasattr(response, "text") and response.text:
            obj = _extract_first_json_object(response.text)
            if obj is not None:
                return obj, None

        # 3) Fallback: try candidates parts
        try:
            candidates = getattr(response, "candidates", []) or []
            for c in candidates:
                parts = getattr(getattr(c, "content", None), "parts", []) or []
                for p in parts:
                    p_text = getattr(p, "text", None)
                    if p_text:
                        obj = _extract_first_json_object(p_text)
                        if obj is not None:
                            return obj, None
        except Exception:
            pass

        # Nothing worked
        return None, "Model returned content that could not be parsed into valid JSON."

    except Exception as e:
        # Return clear error to UI
        return None, f"An error occurred during extraction: {e}"


# --- Utility Function to Convert JSON to Excel (for app.py) ---

def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    """
    Converts the extracted JSON data into an Excel buffer (in-memory file).
    """
    transaction_list = json_data.get("transaction_data", [])
    df = pd.DataFrame(transaction_list)

    output = io.BytesIO()
    # Use openpyxl (in your dependency list)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Transactions
        df.to_excel(writer, sheet_name="Transactions", index=False)

        # Summary
        metadata = {k: v for k, v in json_data.items() if k != "transaction_data"}
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Field", "Value"])
        metadata_df.to_excel(writer, sheet_name="Summary", index=False)

        # Auto-width (approximate) for openpyxl
        ws_t = writer.sheets["Transactions"]
        for i, col in enumerate(df.columns, start=1):
            max_len = max((df[col].astype(str).str.len().max() if not df.empty else 0), len(col)) + 2
            ws_t.column_dimensions[ws_t.cell(row=1, column=i).column_letter].width = max_len

        ws_s = writer.sheets["Summary"]
        for i, col in enumerate(metadata_df.columns, start=1):
            max_len = max((metadata_df[col].astype(str).str.len().max() if not metadata_df.empty else 0), len(col)) + 2
            ws_s.column_dimensions[ws_s.cell(row=1, column=i).column_letter].width = max_len

    output.seek(0)
    return output
