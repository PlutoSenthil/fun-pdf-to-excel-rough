# gemini.py

import io
import json
import pandas as pd
from typing import List, Optional, Dict, Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- Pydantic Models ---

# 1. Transaction Row Model (6 Columns - Unchanged)
class BankTransactionRow(BaseModel):
    date: str = Field(description="The date of the transaction (e.g., '30/10/2025').")
    reference_or_cheque_no: Optional[str] = Field(None, description="The unique transaction reference number, cheque number, or UPI ID/Ref. Null if not provided.")
    description: str = Field(description="The full narration or transaction details.")
    withdrawal_amount: Optional[float] = Field(None, description="The amount withdrawn/debited. Null if it's a credit. Must be a positive number.")
    credit_amount: Optional[float] = Field(None, description="The amount deposited/credited. Null if it's a debit. Must be a positive number.")
    balance: float = Field(description="The running balance of the account *after* this transaction is processed.")

# 2. Overall Statement Model (Includes Header/Footer Info)
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


# --- Core Function to Interact with Gemini ---

def extract_bank_statement(
    api_key: str,
    model_id: str,
    pdf_file_path: str, # File path/handle from Streamlit
    file_bytes: bytes,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts structured bank statement data from a PDF using the Gemini API.

    Args:
        api_key: Your Google Gemini API key.
        model_id: The Gemini model to use (e.g., 'gemini-2.5-flash').
        pdf_file_path: The name of the uploaded file (for display/output naming).
        file_bytes: The actual bytes of the uploaded PDF file.

    Returns:
        A tuple: (Extracted JSON data as a dict, Error message string)
    """
    try:
        # Initialize the client
        client = genai.Client(api_key=api_key)

        # 1. Upload the file to the Gemini File API
        # Using a context manager ensures the file object is closed properly.
        # We pass the bytes directly via an in-memory file for better Streamlit compatibility.
        pdf_file = client.files.upload(
            file=io.BytesIO(file_bytes),
            display_name=pdf_file_path
        )

        # 2. Create the prompt - IMPROVEMENT for more specific extraction
        PROMPT = (
            "You are an expert financial data extraction bot. "
            "From the uploaded Bank Statement PDF document, meticulously extract "
            "ALL header information, footer summaries, and the COMPLETE list "
            "of transactions from the main table, even if it spans multiple pages. "
            "Adhere strictly to the requested JSON schema. All amounts must be positive floats."
        )

        # 3. Call the Gemini API for structured content
        print(f"Sending request to {model_id} for structured extraction...")
        response = client.models.generate_content(
            model=model_id,
            contents=[PROMPT, pdf_file],
            config=types.GenerateContentConfig(
                # Force the model to return a JSON object matching the Pydantic schema
                response_mime_type="application/json",
                response_schema=ExtractedBankStatement,
            ),
        )

        # 4. Process the response and clean up
        extracted_data = json.loads(response.text)

        # Clean up the uploaded file to free up space (crucial for good practice)
        client.files.delete(name=pdf_file.name)
        print(f"Successfully deleted temporary file: {pdf_file.display_name}")

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