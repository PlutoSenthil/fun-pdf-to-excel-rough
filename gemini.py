import io
import json
import re
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# --- Header Aliases (embedded in prompt to help mapping) ---
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


# --- Pydantic Models ---
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

    # OPTIONAL COUNTS (with fallback computation in code)
    debit_count: Optional[int] = Field(
        None, description="The total number of debit transactions in the statement."
    )
    credit_count: Optional[int] = Field(
        None, description="The total number of credit transactions in the statement."
    )


# ----------------- JSON parsing hardener -----------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()

def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first well-formed top-level JSON object from arbitrary text.
    Handles code fences and extra commentary by scanning for balanced braces.
    """
    cleaned = _strip_code_fences(text)
    whole = _try_json_loads(cleaned)
    if whole is not None:
        return whole

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
                        candidate = cleaned[start : idx + 1]
                        obj = _try_json_loads(candidate)
                        if obj is not None:
                            return obj
    return None


def _compute_counts_if_missing(payload: Dict[str, Any]) -> None:
    """Fill debit_count/credit_count if missing or None."""
    tx = payload.get("transaction_data") or []
    if not isinstance(tx, list):
        return
    if payload.get("debit_count") is None:
        payload["debit_count"] = sum(1 for r in tx if (r or {}).get("withdrawal_amount") not in (None, 0, 0.0))
    if payload.get("credit_count") is None:
        payload["credit_count"] = sum(1 for r in tx if (r or {}).get("credit_amount") not in (None, 0, 0.0))


def _detect_unusual_activity(payload: Dict[str, Any]) -> list:
    """
    Heuristic detection of unusual activity:
    - Negative balances
    - IQR outliers in debits/credits
    - Adjacent reversal pairs (equal/opposite amounts)
    - High-frequency same-day repeats (same description+amount)
    Returns list of alert strings (never raises).
    """
    alerts = []
    tx = payload.get("transaction_data") or []
    if not tx:
        return alerts

    debits, credits, balances = [], [], []
    for i, r in enumerate(tx):
        if not isinstance(r, dict):
            continue
        wd = r.get("withdrawal_amount")
        cr = r.get("credit_amount")
        bal = r.get("balance")
        if isinstance(wd, (int, float)) and wd > 0:
            debits.append((i, wd, r))
        if isinstance(cr, (int, float)) and cr > 0:
            credits.append((i, cr, r))
        if isinstance(bal, (int, float)):
            balances.append((i, bal, r))

    # Negative balance
    neg_rows = [i for i, bal, _ in balances if bal < 0]
    if neg_rows:
        alerts.append(f"Negative balances detected in {len(neg_rows)} row(s): e.g., row {neg_rows[0]+1}.")

    # IQR outliers
    def iqr_outliers(values: List[Tuple[int, float, Dict[str, Any]]], label: str):
        if len(values) < 8:
            return
        sorted_vals = sorted(values, key=lambda t: t[1])
        amounts = [v for _, v, _ in sorted_vals]
        q1_idx = len(amounts) // 4
        q3_idx = (len(amounts) * 3) // 4
        q1 = amounts[q1_idx]
        q3 = amounts[q3_idx]
        iqr = max(q3 - q1, 0)
        threshold = q3 + 1.5 * iqr if iqr > 0 else (q3 * 2.5 if q3 > 0 else 0)
        outs = [(i, v, r) for i, v, r in values if v > threshold and threshold > 0]
        if outs:
            e = outs[0]
            alerts.append(
                f"Unusually large {label} detected (IQR rule): {len(outs)} outlier(s), e.g., row {e[0]+1} amount {e[1]:,.2f}."
            )

    iqr_outliers(debits, "debit(s)")
    iqr_outliers(credits, "credit(s)")

    # Reversal detection (adjacent equal amount opposite signs)
    for i in range(len(tx) - 1):
        r1, r2 = tx[i], tx[i + 1]
        wd1, cr1 = r1.get("withdrawal_amount"), r1.get("credit_amount")
        wd2, cr2 = r2.get("withdrawal_amount"), r2.get("credit_amount")
        amt1 = wd1 if wd1 else (cr1 if cr1 else None)
        amt2 = wd2 if wd2 else (cr2 if cr2 else None)
        if isinstance(amt1, (int, float)) and isinstance(amt2, (int, float)) and abs(amt1 - amt2) < 1e-6:
            if (wd1 and cr2) or (cr1 and wd2):
                alerts.append(f"Possible reversal pair at rows {i+1} and {i+2} (equal opposite amounts).")
                break

    # Frequent same-day repeats of same description & amount (>=4)
    from collections import Counter
    key_counts = Counter()
    examples = {}
    for i, r in enumerate(tx):
        d = str(r.get("date", "")).strip()
        desc = str(r.get("description", "")).strip().lower()
        wd = r.get("withdrawal_amount") or 0.0
        cr = r.get("credit_amount") or 0.0
        amt = wd if wd > 0 else cr
        key = (d, desc, round(float(amt), 2))
        key_counts[key] += 1
        if key not in examples:
            examples[key] = i
    repeated = [(k, c) for k, c in key_counts.items() if c >= 4 and (k[2] or 0) > 0]
    if repeated:
        k, c = max(repeated, key=lambda x: x[1])
        alerts.append(
            f"Repeated same-day transactions detected ({c}×): '{k[1]}' on {k[0]} amount {k[2]:,.2f} (possible splitting/bot)."
        )

    return alerts


# --- Core Function: returns (payload, raw_dump, error) ---
def extract_financial_statement(
    api_key: str,
    model_id: str,
    pdf_file_path: str,  # for display only
    file_bytes: bytes,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts structured financial statement data from a PDF using the Gemini API.
    Returns:
      - payload (dict) on success,
      - raw_dump (dict with raw model outputs for backup),
      - error (str) on failure.
    """
    try:
        if not api_key:
            return None, None, "Missing API key."

        client = genai.Client(api_key=api_key)

        # Inline file bytes; explicit mime_type (do NOT pass filename).
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
            "7) Include total_debit_amount, total_credit_amount, closing_balance, and if available, debit_count and credit_count.\n"
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

        # Collect raw strings for backup
        raw_text = getattr(response, "text", None)
        candidates_texts = []
        try:
            candidates = getattr(response, "candidates", []) or []
            for c in candidates:
                parts = getattr(getattr(c, "content", None), "parts", []) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        candidates_texts.append(t)
        except Exception:
            pass

        raw_dump = {
            "model_id": model_id,
            "response_text": raw_text,
            "candidates_texts": candidates_texts,
        }

        # ---------- Robust parsing ----------
        payload: Optional[Dict[str, Any]] = None

        # 1) Prefer structured parse
        if hasattr(response, "parsed") and response.parsed:
            parsed = response.parsed
            if isinstance(parsed, dict):
                payload = parsed
            elif hasattr(parsed, "model_dump"):
                payload = parsed.model_dump()

        # 2) Try response.text
        if payload is None and raw_text:
            payload = _extract_first_json_object(raw_text)

        # 3) Try candidates
        if payload is None and candidates_texts:
            for txt in candidates_texts:
                payload = _extract_first_json_object(txt)
                if payload:
                    break

        if payload is None:
            return None, raw_dump, "Model returned content that could not be parsed into valid JSON."

        # Optional fallback: compute debit/credit counts if missing
        _compute_counts_if_missing(payload)

        # Detect unusual activity
        try:
            alerts = _detect_unusual_activity(payload)
            if alerts:
                payload["alerts"] = alerts
        except Exception:
            pass

        return payload, raw_dump, None

    except Exception as e:
        return None, None, f"An error occurred during extraction: {e}"


# --- Excel Export ---
def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    """
    Converts the extracted JSON data into an Excel buffer (in-memory file).
    Sheets:
      - Transactions
      - Summary
      - Alerts (if any)
    """
    transaction_list = json_data.get("transaction_data", [])
    df = pd.DataFrame(transaction_list)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Transactions
        df.to_excel(writer, sheet_name="Transactions", index=False)

        # Summary (exclude transactions and alerts for separate handling)
        metadata = {k: v for k, v in json_data.items() if k not in ("transaction_data", "alerts")}
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Field", "Value"])
        metadata_df.to_excel(writer, sheet_name="Summary", index=False)

        # Alerts sheet (optional)
        alerts = json_data.get("alerts", [])
        if isinstance(alerts, list) and alerts:
            alerts_df = pd.DataFrame({"Alerts": alerts})
            alerts_df.to_excel(writer, sheet_name="Alerts", index=False)

        # Auto-width for openpyxl
        ws_t = writer.sheets["Transactions"]
        for i, col in enumerate(df.columns, start=1):
            max_len = max((df[col].astype(str).str.len().max() if not df.empty else 0), len(col)) + 2
            ws_t.column_dimensions[ws_t.cell(row=1, column=i).column_letter].width = max_len

        ws_s = writer.sheets["Summary"]
        for i, col in enumerate(metadata_df.columns, start=1):
            max_len = max((metadata_df[col].astype(str).str.len().max() if not metadata_df.empty else 0), len(col)) + 2
            ws_s.column_dimensions[ws_s.cell(row=1, column=i).column_letter].width = max_len

        if isinstance(alerts, list) and alerts:
            ws_a = writer.sheets["Alerts"]
            ws_a.column_dimensions["A"].width = max((len(str(x)) for x in alerts), default=10) + 2

    output.seek(0)
    return output