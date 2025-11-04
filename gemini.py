import io
import json
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


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

        # Crucial: pass explicit mime_type. Do NOT pass 'filename' – not supported in some SDK versions.
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

        # Safer parsing: prefer parsed -> text -> candidates fallback
        extracted_data: Optional[Dict[str, Any]] = None

        if hasattr(response, "parsed") and response.parsed:
            parsed = response.parsed
            # 'parsed' may be a dict or Pydantic model instance
            if isinstance(parsed, dict):
                extracted_data = parsed
            elif hasattr(parsed, "model_dump"):
                extracted_data = parsed.model_dump()
        if extracted_data is None and hasattr(response, "text") and response.text:
            extracted_data = json.loads(response.text)
        if extracted_data is None:
            # Fallback to candidates (defensive)
            try:
                extracted_data = json.loads(response.candidates[0].content.parts[0].text)
            except Exception:
                return None, "Model returned no JSON content."

        return extracted_data, None

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