# app.py

import streamlit as st
import os
import pandas as pd
from gemini import extract_financial_statement, json_to_excel_buffer
from typing import Dict, Any

# --- Model Configuration for Dropdown ---
# Using the model IDs and descriptions based on the table you provided.
MODEL_CHOICES = {
    "gemini-2.5-flash (1 RPM, 4.37K TPM)": "gemini-2.5-flash",
    "gemini-2.5-pro (0 RPM, 0 TPM)": "gemini-2.5-pro",
    "gemini-2.0-flash (0 RPM, 0 TPM)": "gemini-2.0-flash",
    "gemini-2.5-flash-lite (0 RPM, 0 TPM)": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite (0 RPM, 0 TPM)": "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp (0 RPM, 0 TPM)": "gemini-2.0-flash-exp",
    "learnlm-2.0-flash-experimental (0 RPM, 0 TPM)": "learnlm-2.0-flash-experimental",
}

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="PDF Financial Statement Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Gemini-Powered Financial Statement Extractor")
st.markdown("Upload a **PDF Financial Statement** and use a Gemini model to extract structured transaction data into an Excel file.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # 1. API Key Setup
    st.markdown("### ✅ API Key Setup")
    api_key = None
    
    # Try to load from Streamlit secrets first
    try:
        api_key = st.secrets["GOOGLE_API_KEY_1"]
        st.success("API Key loaded successfully from `st.secrets`.")
    except (KeyError, AttributeError):
        # Fallback to text input
        api_key = st.text_input(
            "Enter your Google Gemini API Key", 
            type="password",
            key="api_key_input",
            help="For Streamlit Cloud, use `secrets.toml` with the key `GOOGLE_API_KEY_1`."
        )
        if api_key:
            st.success("API Key set via input.")
        else:
            st.warning("Please set your API Key to proceed.")

    # 2. Model Selection
    st.markdown("### ⚙️ Select Model")
    selected_model_name_display = st.selectbox(
        "Choose the Gemini Model",
        options=list(MODEL_CHOICES.keys()),
        index=0, # Default to Flash 2.5
        help="The model ID in parentheses reflects the current usage rates in the table you provided."
    )
    # Convert display name back to the model ID for the API call
    MODEL_ID = MODEL_CHOICES[selected_model_name_display]
    st.info(f"Selected Model ID: `{MODEL_ID}`")

    # 3. File Uploader
    st.markdown("### ⬆️ Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF Financial Statement", 
        type=["pdf"]
    )
    
# --- Main Application Logic ---

# Status Check
is_ready = (uploaded_file is not None) and (api_key is not None and api_key != "")

if uploaded_file:
    # Display file information after upload
    st.subheader("Uploaded File Details")
    col1, col2 = st.columns(2)
    col1.metric("File Name", uploaded_file.name)
    col2.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")

    st.markdown("---")
    if not is_ready:
        st.error("Please complete the API Key and PDF upload steps in the sidebar.")
    else:
        start_button = st.button("🚀 Start Data Extraction", type="primary")

        if start_button:
            # Use a try-except block for the main process
            try:
                # Display a progress indicator
                with st.spinner(f"Extracting data using **{MODEL_ID}**... This may take a moment."):
                    
                    # Read the file into bytes
                    uploaded_file.seek(0) # Ensure we read from the start of the file buffer
                    file_bytes = uploaded_file.read()
                    
                    # Get the extracted JSON data and any potential error
                    extracted_data, error = extract_financial_statement(
                        api_key=api_key,
                        model_id=MODEL_ID,
                        pdf_file_path=uploaded_file.name,
                        file_bytes=file_bytes
                    )

                if error:
                    # Show the error message from the core function
                    st.error(f"Extraction Failed: {error}")
                
                elif extracted_data:
                    st.success("🎉 Data Extraction Complete!")

                    # 1. Display Header/Summary Data
                    st.subheader("Summary Information")
                    
                    summary_cols = ['institution_name', 'account_holder_name', 'statement_period', 'initial_balance', 'closing_balance']
                    summary_data = {k.replace('_', ' ').title(): extracted_data.get(k) for k in summary_cols}
                    
                    summary_df = pd.DataFrame(summary_data.items(), columns=["Field", "Value"])
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)


                    # 2. Display Transaction Data (First 10 rows preview)
                    st.subheader("Transaction Data Preview")
                    transaction_list = extracted_data.get("transaction_data", [])
                    if transaction_list:
                        df_preview = pd.DataFrame(transaction_list)
                        st.dataframe(df_preview.head(10), use_container_width=True)
                        st.info(f"Successfully extracted **{len(transaction_list)}** transactions. Download the full file below.")

                        # 3. Create and offer the Excel file for download
                        excel_buffer = json_to_excel_buffer(extracted_data)
                        
                        # Determine the output filename (match input)
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        output_filename = f"{base_name}_extracted.xlsx"

                        st.download_button(
                            label="⬇️ Download Extracted Data as Excel (.xlsx)",
                            data=excel_buffer,
                            file_name=output_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("The model returned summary data but no transactions were found in the PDF. Please try a different model.")
                
            except Exception as e:
                # Catch any unexpected application-level errors
                st.error(f"An unexpected application error occurred: {e}")