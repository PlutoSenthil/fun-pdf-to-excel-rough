# app.py

import streamlit as st
import os
from gemini import extract_bank_statement, json_to_excel_buffer
from typing import Dict, Any

# --- Model Configuration for Dropdown ---
# Key: Display Name | Value: Model ID (for API)
MODEL_CHOICES = {
    "Gemini 2.5 Flash (Recommended for Extraction)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (Highest Accuracy)": "gemini-2.5-pro",
    "Gemini 2.0 Flash (Legacy)": "gemini-2.0-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    # Note: I am simplifying the complex table info you provided into a more UI-friendly format.
}
# The complex table with RPM/TPM info is not user-selectable, but informs the choices.

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="PDF Bank Statement Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Gemini-Powered Bank Statement Extractor")
st.markdown("Upload a **Bank Statement PDF** and use a Gemini model to extract structured transaction data into an Excel file.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # 1. API Key from Streamlit Secrets
    st.markdown("1. **API Key Setup**")
    try:
        # Access the secret key from Streamlit secrets
        api_key = st.secrets["GOOGLE_API_KEY_1"]
        st.success("API Key loaded from `st.secrets`.")
    except (KeyError, AttributeError):
        api_key = st.text_input(
            "Enter your Google Gemini API Key", 
            type="password",
            help="For Streamlit Cloud, use `secrets.toml` with the key `GOOGLE_API_KEY_1`."
        )
        if not api_key:
            st.warning("Please set your API Key to proceed.")

    # 2. Model Selection
    st.markdown("2. **Select Model**")
    selected_model_name = st.selectbox(
        "Choose the Gemini Model",
        options=list(MODEL_CHOICES.keys()),
        index=0, # Default to Flash
        help="2.5 Flash is usually sufficient and cost-effective. Use 2.5 Pro for maximum accuracy on complex or messy PDFs."
    )
    # Convert display name back to the model ID for the API call
    MODEL_ID = MODEL_CHOICES[selected_model_name]
    st.info(f"Using Model ID: `{MODEL_ID}`")

    # 3. File Uploader
    st.markdown("3. **Upload PDF**")
    uploaded_file = st.file_uploader(
        "Upload a PDF Bank Statement", 
        type=["pdf"]
    )

# --- Main Application Logic ---

if uploaded_file and api_key:
    # Use a try-except block for the main process
    try:
        # Display a progress indicator
        with st.spinner(f"Extracting data using {MODEL_ID}... This may take a moment."):
            
            # Read the file into bytes
            file_bytes = uploaded_file.read()
            
            # Get the extracted JSON data and any potential error
            extracted_data, error = extract_bank_statement(
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
            
            # Create a simple two-column layout for the summary
            summary_cols = ['bank_name', 'account_holder_name', 'statement_period', 'initial_balance', 'closing_balance']
            summary_data = {k.replace('_', ' ').title(): extracted_data.get(k) for k in summary_cols}
            st.dataframe(summary_data.items(), use_container_width=True, hide_index=True)


            # 2. Display Transaction Data (First 10 rows preview)
            st.subheader("Transaction Data Preview")
            transaction_list = extracted_data.get("transaction_data", [])
            if transaction_list:
                df_preview = pd.DataFrame(transaction_list)
                st.dataframe(df_preview.head(10), use_container_width=True)
                st.info(f"Successfully extracted **{len(transaction_list)}** transactions.")

                # 3. Create and offer the Excel file for download
                excel_buffer = json_to_excel_buffer(extracted_data)
                
                # Determine the output filename (match input)
                base_name = os.path.splitext(uploaded_file.name)[0]
                output_filename = f"{base_name}.xlsx"

                st.download_button(
                    label="⬇️ Download Extracted Data as Excel (.xlsx)",
                    data=excel_buffer,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("The model returned summary data but no transactions were found in the PDF.")
        
    except Exception as e:
        # Catch any unexpected application-level errors
        st.error(f"An unexpected application error occurred: {e}")

elif uploaded_file and not api_key:
    st.warning("Please enter or set your Google Gemini API Key in the sidebar.")
elif not uploaded_file and api_key:
    st.info("Please upload a PDF Bank Statement in the sidebar to begin.")