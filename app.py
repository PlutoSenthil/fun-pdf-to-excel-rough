import os
import pandas as pd
import streamlit as st
from gemini import extract_financial_statement, json_to_excel_buffer

# --- Config ---
MODEL_CHOICES = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.0 Flash": "gemini-2.0-flash",  # default
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}

st.set_page_config(page_title="PDF Statement Extractor", layout="wide")
st.title("📄 Gemini-Powered Financial Statement Extractor")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY_1"]
        st.success("API Key loaded from secrets.")
    except Exception:
        api_key = st.text_input("Enter Google Gemini API Key", type="password")
    model_name = st.selectbox("Select Model", list(MODEL_CHOICES.keys()), index=1)
    MODEL_ID = MODEL_CHOICES[model_name]
    uploaded_file = st.file_uploader("Upload PDF Statement", type=["pdf"])

# --- Main ---
if uploaded_file and api_key:
    st.subheader("Uploaded File")
    st.write(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size/1024:.1f} KB | **Model:** {MODEL_ID}")

    if st.button("🚀 Start Data Extraction"):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        with st.spinner("Extracting data..."):
            data, error = extract_financial_statement(api_key, MODEL_ID, file_bytes)

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
                                   file_name=f"{os.path.splitext(uploaded_file.name)[0]}_extracted.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No transactions found.")