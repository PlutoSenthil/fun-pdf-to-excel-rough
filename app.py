import os
import pandas as pd
import streamlit as st
from gemini import extract_financial_statement, json_to_excel_buffer
from typing import Dict, Any, Optional, List, Tuple
# --- Config ---
MODEL_CHOICES = {
    "Gemini 2.5 Flash": "gemini-2.5-flash", # default
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2.0 Flash": "gemini-2.0-flash",  
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash Experimental": "gemini-2.0-flash-exp",
    "LearnLM 2.0 Flash Experimental": "learnlm-2.0-flash-experimental",
}
st.set_page_config(page_title="PDF Statement Extractor", layout="wide")
st.title("📄 Gemini-Powered Financial Statement Extractor")

# ---------------- Helpers ----------------
def _mask_key(k: str) -> str:
    if not k or len(k) < 8:
        return "********"
    return f"{k[:4]}...{k[-4:]}"


def _load_available_api_keys() -> List[Tuple[str, str]]:
    found = []
    for i in (1, 2, 3):
        name = f"GOOGLE_API_KEY_{i}"
        try:
            val = st.secrets[name]
            if val:
                found.append((name, val))
        except Exception:
            continue
    return found

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API Key selector
    st.markdown("### ✅ API Key")
    available_keys = _load_available_api_keys()
    api_key_label: str = "MANUAL"
    api_key: Optional[str] = None

    if available_keys:
        labels = [f"{name} ({_mask_key(val)})" for name, val in available_keys]
        default_idx = 0  # defaults to GOOGLE_API_KEY_1
        sel_idx = st.selectbox(
            "Select API Key",
            options=range(len(labels)),
            index=default_idx,
            format_func=lambda i: labels[i],
        )
        api_key_label, api_key = available_keys[sel_idx]
        st.caption(f"Using API Key: **{_mask_key(api_key)}**")
    else:
        api_key = st.text_input(
            "Enter your Google Gemini API Key",
            type="password",
            key="api_key_input",
        )
        if api_key:
            st.caption(f"Using API Key: **{_mask_key(api_key)}**")
        else:
            st.warning("Please provide an API key to proceed.")

    # Model selection
    st.markdown("### ⚙️ Model")
    options = list(MODEL_CHOICES.keys())
    default_model_idx = options.index("Gemini 2.5 Flash") if "Gemini 2.5 Flash" in options else 0
    model_display = st.selectbox("Choose the Gemini Model", options=options, index=default_model_idx)
    MODEL_ID = MODEL_CHOICES[model_display]
    st.info(f"Selected Model ID: `{MODEL_ID}`")

    # File upload
    st.markdown("### ⬆️ Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF Financial Statement", type=["pdf"])

# --- Main ---
if uploaded_file and api_key:
    st.subheader("Uploaded File")
    st.write(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size/1024:.1f} KB | **Model:** {MODEL_ID}")

    if st.button("🚀 Start Data Extraction"):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        with st.spinner("Extracting data..."):
            # CORRECTED LINE: passing file_bytes as a keyword argument
            data, error = extract_financial_statement(api_key, MODEL_ID, file_bytes=file_bytes)

        if error:
            st.error(error)
        elif data:
            st.success("✅ Done.")
            # Summary
            st.subheader("Summary")
            summary_cols = ["institution_name", "account_holder_name", "statement_period",
                            "initial_balance", "closing_balance", "total_debit_amount", "total_credit_amount"]
            summary_data = {k.replace("_", " ").title(): data.get(k) for k in summary_cols}
            st.dataframe(pd.DataFrame(summary_data.items(), columns=["Field", "Value"]), use_container_width=True)

            # Transactions
            st.subheader("Transactions Preview")
            tx = data.get("transaction_data", [])
            if tx:
                df = pd.DataFrame(tx)
                st.dataframe(df.head(10), use_container_width=True)
                excel_buffer = json_to_excel_buffer(data)
                st.download_button("⬇️ Download Excel", data=excel_buffer,
                                   file_name=f"{os.path.splitext(uploaded_file.name)[0]}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No transactions found.")