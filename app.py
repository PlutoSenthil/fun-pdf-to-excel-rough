import os
import time
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import streamlit as st

from gemini import extract_financial_statement, json_to_excel_buffer


# ---------------- Essentials: Model list ----------------
MODEL_CHOICES = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash",  # default
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
    "Gemini 2.0 Flash Experimental": "gemini-2.0-flash-exp",
    "LearnLM 2.0 Flash Experimental": "learnlm-2.0-flash-experimental",
}

st.set_page_config(
    page_title="PDF Financial Statement Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Financial Statement Extractor (Gemini)")
st.caption("Essentials only: API Key → Model → Upload PDF → Start Extraction → Summary → Transactions Preview → Excel download")


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


# ---------------- Sidebar (Essentials only) ----------------
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
    default_model_idx = options.index("Gemini 2.0 Flash") if "Gemini 2.0 Flash" in options else 0
    model_display = st.selectbox("Choose the Gemini Model", options=options, index=default_model_idx)
    MODEL_ID = MODEL_CHOICES[model_display]
    st.info(f"Selected Model ID: `{MODEL_ID}`")

    # File upload
    st.markdown("### ⬆️ Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF Financial Statement", type=["pdf"])


# ---------------- Main (Essentials only flow) ----------------
is_ready = (uploaded_file is not None) and (api_key is not None and api_key.strip() != "")

if uploaded_file:
    st.subheader("Uploaded File")
    c1, c2, c3 = st.columns(3)
    c1.metric("File Name", uploaded_file.name)
    c2.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
    c3.metric("Model", MODEL_ID)

    st.markdown("---")

    if not is_ready:
        st.error("Please complete API Key and file upload.")
    else:
        if st.button("🚀 Start Extraction", type="primary"):
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()

                t0 = time.time()
                with st.spinner(f"Extracting with **{MODEL_ID}**..."):
                    extracted_data, raw_text, error = extract_financial_statement(
                        api_key=api_key,
                        model_id=MODEL_ID,
                        pdf_file_path=uploaded_file.name,
                        file_bytes=file_bytes,
                    )
                _ = time.time() - t0  # duration not displayed (essentials only)

                if error:
                    st.error("Extraction failed.")
                    # Essentials-only fallback: let user download raw text (no salvage parsing here)
                    if raw_text:
                        base = os.path.splitext(uploaded_file.name)[0]
                        st.download_button(
                            "⬇️ Download raw Gemini output (.txt)",
                            data=raw_text.encode("utf-8", errors="ignore"),
                            file_name=f"{base}_gemini_raw.txt",
                            mime="text/plain",
                        )
                else:
                    st.success("✅ Done.")

                    # Summary
                    st.subheader("Summary")
                    summary_cols = [
                        "institution_name",
                        "account_holder_name",
                        "statement_period",
                        "initial_balance",
                        "closing_balance",
                        "total_debit_amount",
                        "total_credit_amount",
                        "debit_count",
                        "credit_count",
                    ]
                    summary_map = {k.replace("_", " ").title(): extracted_data.get(k) for k in summary_cols}
                    summary_df = pd.DataFrame(summary_map.items(), columns=["Field", "Value"])
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    # Transactions preview
                    st.subheader("Transactions Preview")
                    tx = extracted_data.get("transaction_data", [])
                    if tx:
                        df_preview = pd.DataFrame(tx)
                        st.dataframe(df_preview.head(10), use_container_width=True)
                        st.info(f"Extracted **{len(tx)}** transactions. Download the full file below.")

                        # Excel download
                        excel_buffer = json_to_excel_buffer(extracted_data)
                        base = os.path.splitext(uploaded_file.name)[0]
                        st.download_button(
                            "⬇️ Download Extracted Data (.xlsx)",
                            data=excel_buffer,
                            file_name=f"{base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        st.warning("No transactions found.")
            except Exception:
                st.error("Unexpected application error.")