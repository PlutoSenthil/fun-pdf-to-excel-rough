import io
import json
import re
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ----------------- Header Aliases (prompt hint) -----------------
HEADER_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date", "posting date"],
    "reference_or_cheque_no": [
        "ref", "reference", "cheque no", "cheque no.", "chq no", "chq/ref",
        "utr", "upi ref", "upi id", "rrn", "transaction id", "neft ref", "imps ref"
    ],
    "description": ["description", "narration", "details", "particulars", "remarks"],
    "withdrawal_amount": ["withdrawal", "withdrawals", "debit", "dr", "debit amount", "withdrawal amount"],
    "credit_amount": ["deposit", "credit", "cr", "credit amount", "deposit amount"],
    "balance": ["balance", "running balance", "closing balance", "bal"]
}


# ----------------- Pydantic Schema -----------------
class FinancialTransactionRow(BaseModel):
    date: str = Field(description="The date of the transaction (e.g., '30/10/2025').")
    reference_or_cheque_no: Optional[str] = Field(
        None, description="Reference number, cheque number, or UPI ID/Ref. Null if not provided."
    )
    description: str = Field(description="The full narration or transaction details.")
    withdrawal_amount: Optional[float] = Field(
        None, description="Debited amount; positive number or null if credit."
    )
    credit_amount: Optional[float] = Field(
        None, description="Credited amount; positive number or null if debit."
    )
    balance: float = Field(description="Running balance after this transaction.")


class ExtractedFinancialStatement(BaseModel):
    # Header
    institution_name: str = Field(description="Financial institution name.")
    branch_name: str = Field(description="Branch name.")
    account_holder_name: str = Field(description="Primary account holder.")
    statement_period: str = Field(description="e.g., '01/10/2025 to 31/10/2025'.")
    initial_balance: float = Field(description="Opening balance at period start.")

    # Transactions
    transaction_data: List[FinancialTransactionRow]

    # Footer/Summary
    total_debit_amount: float = Field(description="Sum of all withdrawal amounts.")
    total_credit_amount: float = Field(description="Sum of all credit amounts.")
    closing_balance: float = Field(description="Final balance at period end.")

    # Optional counts (kept optional; we will compute if missing)
    debit_count: Optional[int] = Field(
        None, description="Total number of debit transactions."
    )
    credit_count: Optional[int] = Field(
        None, description="Total number of credit transactions."
    )


# ----------------- JSON Salvage Utilities -----------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_CTRL_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # disallowed in JSON strings


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()


def _clean_json_text(s: str) -> str:
    if not s:
        return s
    s = s.replace("\ufeff", "")  # BOM
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")  # smart quotes
    s = _CTRL_CHARS_RE.sub("", s)  # control chars
    return s.strip()


def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _remove_trailing_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    return re.sub(r',(\s*[}\]])', r'\1', s)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first well-formed top-level JSON object from arbitrary text.
    Strategy: clean -> try as-is -> try without trailing commas -> scan for balanced braces.
    """
    cleaned = _strip_code_fences(_clean_json_text(text))

    whole = _try_json_loads(cleaned)
    if whole is not None:
        return whole

    rc = _try_json_loads(_remove_trailing_commas(cleaned))
    if rc is not None:
        return rc

    # Scan for first balanced object
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in starts:
        depth, in_str, esc = 0, False, False
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
                        candidate = cleaned[start: idx + 1]
                        obj = _try_json_loads(candidate) or _try_json_loads(_remove_trailing_commas(candidate))
                        if obj is not None:
                            return obj
    # Optional: lenient parse via json5 if present (not required)
    try:
        import json5  # type: ignore
        try:
            return json5.loads(cleaned)
        except Exception:
            pass
    except Exception:
        pass
    return None


def _compute_counts_if_missing(payload: Dict[str, Any]) -> None:
    """Compute debit_count/credit_count if model omitted them."""
    tx = payload.get("transaction_data") or []
    if payload.get("debit_count") is None:
        payload["debit_count"] = sum(1 for r in tx if (r or {}).get("withdrawal_amount") not in (None, 0, 0.0))
    if payload.get("credit_count") is None:
        payload["credit_count"] = sum(1 for r in tx if (r or {}).get("credit_amount") not in (None, 0, 0.0))


# ----------------- Core API -----------------
def extract_financial_statement(
    api_key: str,
    model_id: str,
    pdf_file_path: str,
    file_bytes: bytes,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract structured bank statement data with Gemini.
    Returns (payload_dict, error_message).
    """
    try:
        if not api_key:
            return None, "Missing API key."

        client = genai.Client(api_key=api_key)

        # Inline bytes (explicit mime type). Do NOT pass filename (not supported in some SDK versions).
        pdf_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf",
        )

        PROMPT = (
            "You are validating bank statement table rows and returning a structured JSON that matches the provided schema.\n"
            "Rules:\n"
            "1) Do not fabricate values.\n"
            "2) Keep date formats as-is.\n"
            "3) Ensure correct withdrawal_amount and credit_amount per row (exactly one is non-null when applicable).\n"
            "4) Use numeric types for amounts and balance.\n"
            "5) The running balance is the balance after each transaction.\n"
            "6) Map headers using these aliases where needed:\n"
            f"   {json.dumps(HEADER_ALIASES)}\n"
            "7) Include total_debit_amount, total_credit_amount, closing_balance; include debit_count and credit_count when available.\n"
            "8) Return JSON only (no markdown, no code fences, no commentary).\n"
            "9) If a field is not present in the PDF, use empty string for text fields and null for numeric fields.\n"
            "10) Accuracy is critical; do not infer values that aren't present.\n"
        )

        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, pdf_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFinancialStatement,
            ),
        )

        # Prefer structured parse -> salvage from response.text -> salvage from candidates
        payload: Optional[Dict[str, Any]] = None

        if hasattr(response, "parsed") and response.parsed:
            parsed = response.parsed
            if isinstance(parsed, dict):
                payload = parsed
            elif hasattr(parsed, "model_dump"):
                payload = parsed.model_dump()

        if payload is None and hasattr(response, "text") and response.text:
            payload = _extract_first_json_object(response.text)

        if payload is None:
            try:
                candidates = getattr(response, "candidates", []) or []
                for c in candidates:
                    parts = getattr(getattr(c, "content", None), "parts", []) or []
                    for p in parts:
                        txt = getattr(p, "text", None)
                        if txt:
                            payload = _extract_first_json_object(txt)
                            if payload:
                                break
                    if payload:
                        break
            except Exception:
                pass

        if payload is None:
            return None, "Could not parse JSON response."

        _compute_counts_if_missing(payload)
        return payload, None

    except Exception:
        return None, "Extraction error."


# ----------------- Excel Export -----------------
def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    """
    Writes:
      - Sheet 'Transactions': full transaction_data
      - Sheet 'Summary': all non-table fields
    """
    transaction_list = json_data.get("transaction_data", [])
    df = pd.DataFrame(transaction_list)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Transactions
        df.to_excel(writer, sheet_name="Transactions", index=False)

        # Summary (all keys except the table)
        metadata = {k: v for k, v in json_data.items() if k != "transaction_data"}
        meta_df = pd.DataFrame(list(metadata.items()), columns=["Field", "Value"])
        meta_df.to_excel(writer, sheet_name="Summary", index=False)

        # Column widths (approx)
        ws_t = writer.sheets["Transactions"]
        for i, col in enumerate(df.columns, start=1):
            max_len = max((df[col].astype(str).str.len().max() if not df.empty else 0), len(col)) + 2
            ws_t.column_dimensions[ws_t.cell(row=1, column=i).column_letter].width = max_len

        ws_s = writer.sheets["Summary"]
        for i, col in enumerate(meta_df.columns, start=1):
            max_len = max((meta_df[col].astype(str).str.len().max() if not meta_df.empty else 0), len(col)) + 2
            ws_s.column_dimensions[ws_s.cell(row=1, column=i).column_letter].width = max_len

    output.seek(0)
    return output