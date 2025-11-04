import io
import json
import re
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ----------------- Schema Definitions -----------------

class FinancialTransactionRow(BaseModel):
    """Schema for a single transaction row in the bank statement."""
    date: str = Field(description="The date of the transaction, kept in its original format.")
    reference_or_cheque_no: Optional[str] = Field(None, description="Reference or cheque number, if available.")
    description: str = Field(description="Detailed description of the transaction.")
    withdrawal_amount: Optional[float] = Field(None, description="The amount withdrawn (debit), if applicable. Should be a positive number.")
    credit_amount: Optional[float] = Field(None, description="The amount credited, if applicable. Should be a positive number.")
    balance: float = Field(description="The running balance after the transaction.")

class ExtractedFinancialStatement(BaseModel):
    """Overall schema for the extracted financial statement data."""
    institution_name: str = Field(description="Name of the financial institution (Bank name).")
    branch_name: str = Field(description="Name or location of the branch.")
    account_holder_name: str = Field(description="The name of the account holder.")
    statement_period: str = Field(description="The period covered by the statement (e.g., 'Jan 1, 2023 - Jan 31, 2023').")
    initial_balance: float = Field(description="The opening balance at the start of the statement period.")
    transaction_data: List[FinancialTransactionRow] = Field(description="A list of all individual financial transactions.")
    total_debit_amount: float = Field(description="The sum of all withdrawal_amount values in the statement.")
    total_credit_amount: float = Field(description="The sum of all credit_amount values in the statement.")
    closing_balance: float = Field(description="The closing balance at the end of the statement period.")
    debit_count: Optional[int] = Field(None, description="The total number of debit transactions.")
    credit_count: Optional[int] = Field(None, description="The total number of credit transactions.")

# ----------------- JSON Salvage Functions -----------------
# These are included to help parse the response text if the built-in
# JSON parsing (response.parsed) fails, which is a common robustness
# technique when working with LLM outputs.

def _clean_json_text(s: str) -> str:
    """Removes common LLM-related JSON artifacts like backticks and BOM."""
    # Remove Byte Order Mark (BOM)
    s = s.replace("\ufeff", "")
    # Remove markdown code block delimiters (```json or ```)
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.MULTILINE)
    return s

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Tries to load JSON from cleaned text, falling back to substring search."""
    cleaned = _clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # Try to find first balanced braces as a salvage mechanism
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                # Attempt to load JSON from the substring
                return json.loads(cleaned[start:end+1])
            except Exception:
                # Still failed
                return None
    return None

# ----------------- Core Extraction Logic -----------------

def extract_financial_statement(api_key: str, model_id: str, file_bytes: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Uses the Gemini API to extract financial statement data from a PDF byte array.

    Args:
        api_key: Your Google Gemini API key.
        model_id: The ID of the Gemini model to use.
        file_bytes: The byte content of the PDF file.

    Returns:
        A tuple: (Extracted data dictionary, Error message string)
    """
    try:
        # 1. Setup Client and PDF Part
        client = genai.Client(api_key=api_key)
        # Create a Part object from the PDF bytes
        pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")

        # 2. Define Extraction Prompt
        PROMPT = (
            "Extract all header info, transaction rows, and summary from this bank statement PDF.\n"
            "Rules:\n"
            "- Do not fabricate values.\n"
            "- Keep date formats as-is.\n"
            "- Ensure correct withdrawal_amount or credit_amount per row. One must be None, the other a float.\n"
            "- Use numeric types for amounts and balance (float).\n"
            "- The transaction_data must be a complete list of all transactions found.\n"
            "- Return JSON only matching the provided schema."
        )

        # 3. Generate Content with Schema Enforcement
        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, pdf_part],
            config=types.GenerateContentConfig(
                # Enforce JSON output type
                response_mime_type="application/json",
                # Enforce structure via the Pydantic schema
                response_schema=ExtractedFinancialStatement,
            ),
        )

        # 4. Parse Response (Prioritize built-in parsing)
        if hasattr(response, "parsed") and response.parsed:
            # The SDK successfully parsed the JSON into the Pydantic model
            return response.parsed.model_dump(), None

        if hasattr(response, "text") and response.text:
            # Fallback: The SDK didn't parse, try manual salvage
            data = _extract_json(response.text)
            if data:
                # Optional: Re-validate against Pydantic model here for extra safety
                try:
                    ExtractedFinancialStatement.model_validate(data)
                    return data, None
                except Exception as ve:
                    # Return the raw JSON even if validation fails, as a best effort
                    return data, f"Warning: Data extracted but failed final schema validation. Error: {ve}"
            else:
                # Include the raw text in the error for debugging
                return None, f"Could not parse JSON. Raw LLM output (first 250 chars): {response.text[:250]}..."

        return None, "Empty response from API."

    except Exception as e:
        return None, f"An API or client error occurred: {e}"

# ----------------- Excel Export Logic -----------------

def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    """
    Converts the extracted financial data dictionary into an Excel file buffer.

    Args:
        json_data: The dictionary containing the extracted financial data.

    Returns:
        An io.BytesIO object containing the Excel file content.
    """
    # 1. Transactions Sheet
    df = pd.DataFrame(json_data.get("transaction_data", []))

    # 2. Summary Sheet
    # Filter out the transaction list for the summary sheet
    summary = {k: v for k, v in json_data.items() if k != "transaction_data"}
    summary_df = pd.DataFrame(list(summary.items()), columns=["Field", "Value"])

    # 3. Create Excel Writer and Write Data to Buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Transactions", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)
    return output