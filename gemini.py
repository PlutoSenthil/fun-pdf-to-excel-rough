# gemini.py

import io
import json
import pandas as pd
from typing import List, Optional, Dict, Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- Pydantic Models ---
# Note: Renamed models to avoid the word 'Bank'

# 1. Transaction Row Model (6 Columns)
class FinancialTransactionRow(BaseModel):
    date: str = Field(description="The date of the transaction (e.g., '30/10/2025').")
    reference_or_cheque_no: Optional[str] = Field(None, description="The unique transaction reference number, cheque number, or UPI ID/Ref. Null if not provided.")
    description: str = Field(description="The full narration or transaction details.")
    withdrawal_amount: Optional[float] = Field(None, description="The amount withdrawn/debited. Null if it's a credit. Must be a positive number.")
    credit_amount: Optional[float] = Field(None, description="The amount deposited/credited. Null if it's a debit. Must be a positive number.")
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
    pdf_file_path: str, # File path/handle from Streamlit
    file_bytes: bytes,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts structured financial statement data from a PDF using the Gemini API 
    via the reliable Part.from_bytes() method (inline file transfer).
    NOTE: This method is best for files under 20MB.
    """
    try:
        # Initialize the client
        client = genai.Client(api_key=api_key)

        # 1. Define the PDF content part using bytes and explicit mime_type.
        # This bypasses client.files.upload() which was causing the MIME type/keyword argument issues.
        pdf_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf"
        )

        # 2. Create the prompt 
        PROMPT = (
            "You are an expert financial data extraction bot. "
            "From the uploaded Financial Statement PDF document, meticulously extract "
            "ALL header information (Institution name, holder name, period, initial balance), "
            "footer summaries (total debit/credit, closing balance), and the COMPLETE list "
            "of transactions from the main table, even if it spans multiple pages. "
            "Adhere strictly to the requested JSON schema. All amounts must be positive floats."
        )

        # 3. Call the Gemini API for structured content
        print(f"Sending request to {model_id} for structured extraction...")
        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, pdf_part], # Contents now include the prompt text and the PDF bytes part
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFinancialStatement,
            ),
        )

        # 4. Process the response. (No cleanup needed as the file was sent inline)
        extracted_data = json.loads(response.text)

        return extracted_data, None

    except Exception as e:
        # Catch and return any API or JSON parsing errors
        return None, f"An error occurred during extraction: {e}"

# --- Utility Function to Convert JSON to Excel (for app.py) ---

def json_to_excel_buffer(json_data: Dict[str, Any]) -> io.BytesIO:
    """
    Converts the extracted JSON data into an Excel buffer (in-memory file).
    """
    # 1. Create the DataFrame from the 'transaction_data' list
    transaction_list = json_data.get("transaction_data", [])
    df = pd.DataFrame(transaction_list)
    
    # 2. Create an in-memory buffer
    output = io.BytesIO()
    
    # 3. Use an ExcelWriter to write data and metadata to different sheets
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write Transaction Data to the first sheet
        df.to_excel(writer, sheet_name='Transactions', index=False)

        # Write Header/Summary Data to the second sheet (Metadata)
        metadata = {k: v for k, v in json_data.items() if k != 'transaction_data'}
        
        # Convert dictionary to a two-column DataFrame for a clean look
        metadata_df = pd.DataFrame(list(metadata.items()), columns=['Field', 'Value'])
        metadata_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Auto-adjust column widths for readability
        worksheet_t = writer.sheets['Transactions']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).str.len().max() if not df[col].empty else 0, len(col)) + 2
            worksheet_t.set_column(i, i, max_len)

        worksheet_s = writer.sheets['Summary']
        for i, col in enumerate(metadata_df.columns):
            max_len = max(metadata_df[col].astype(str).str.len().max() if not metadata_df[col].empty else 0, len(col)) + 2
            worksheet_s.set_column(i, i, max_len)

    # Move the buffer's cursor to the beginning
    output.seek(0)
    return output