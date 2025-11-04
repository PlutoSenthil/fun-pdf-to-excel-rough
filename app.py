import os
from typing import Dict, Any

import pandas as pd
import streamlit as st

from gemini import extract_financial_statement, json_to_excel_buffer

# --- Model Configuration for Dropdown ---
# Default should be "Gemini 2.0 Flash": "gemini-2.0-flash"
MODEL_CHOICES = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash",  # default choice
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
    "Gemini 2.0 Flash Experimental": "gemini-2.0-flash-exp",
    "LearnLM 2.0 Flash Experimental": "learnlm-2.0-flash-experimental",
}

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="PDF Financial Statement Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Gemini-Powered Financial Statement Extractor")
st.markdown(
    "Upload a **PDF Financial Statement** and use a Gemini model to extract structured transaction data into an Excel file."
)

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")

    # 1. API Key Setup
    st.markdown("### ✅ API Key Setup")
    api_key = None

    # Try to load from Streamlit secrets first
    loaded_from_secrets = False
    try:
        api_key = st.secrets["GOOGLE_API_KEY_1"]
        loaded_from_secrets = True
        st.success("API Key loaded from `st.secrets`.")
    except (KeyError, AttributeError):
        # Fallback to text input (hidden)
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

    # Masked API key preview (first 4 + last 4 chars) so you can verify the key without exposing it
    def _mask_key(k: str) -> str:
        if not k or len(k) < 8:
            return "********"
        return f"{k[:4]}...{k[-4:]}"

    if api_key:
        source_label = "secrets" if loaded_from_secrets else "input"
        st.caption(f"Using API Key ({source_label}): **{_mask_key(api_key)}**")

    # 2. Model Selection
    st.markdown("### ⚙️ Select Model")
    options = list(MODEL_CHOICES.keys())
    default_index = options.index("Gemini 2.0 Flash") if "Gemini 2.0 Flash" in options else 0

    selected_model_name_display = st.selectbox(
        "Choose the Gemini Model",
        options=options,
        index=default_index,
        help="Flash = fast & cost-effective; Pro = higher quality but may be slower or restricted."
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
is_ready = (uploaded_file is not None) and (api_key is not None and api_key.strip() != "")

if uploaded_file:
    # Display file information after upload
    st.subheader("Uploaded File Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("File Name", uploaded_file.name)
    c2.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
    c3.metric("Model Selected", MODEL_ID)  # Show which model will process this file

    st.markdown("---")
    if not is_ready:
        st.error("Please complete the API Key and PDF upload steps in the sidebar.")
    else:
        start_button = st.button("🚀 Start Data Extraction", type="primary")

        if start_button:
            try:
                with st.spinner(f"Extracting data using **{MODEL_ID}**... This may take a moment."):
                    # Read the file into bytes
                    uploaded_file.seek(0)  # Ensure we read from the start of the file buffer
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

                    # A few possible causes to guide troubleshooting (non-intrusive hints)
                    with st.expander("Troubleshooting tips"):
                        st.markdown(
                            "- Ensure the API key is valid and has access to the selected model.\n"
                            "- Try a text-based (non-scanned) PDF under ~20MB.\n"
                            "- If the PDF is scanned, try a clearer copy; OCR quality matters.\n"
                            "- Try another model (e.g., Gemini 2.5 Flash vs 2.0 Flash).\n"
                            "- If the error mentions JSON parsing: the model returned extra text—please retry once."
                        )

                elif extracted_data:
                    st.success("🎉 Data Extraction Complete!")

                    # 1. Display Header/Summary Data
                    st.subheader("Summary Information")

                    summary_cols = [
                        'institution_name', 'account_holder_name',
                        'statement_period', 'initial_balance', 'closing_balance'
                    ]
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
                        st.warning(
                            "The model returned summary data but no transactions were found in the PDF. "
                            "Try a different model or ensure the PDF contains a clearly formatted transaction table."
                        )

            except Exception as e:
                st.error(f"An unexpected application error occurred: {e}")