import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import streamlit as st

from gemini import extract_financial_statement, json_to_excel_buffer


# ---------------- Model & Key Selection ----------------
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
    initial_sidebar_state="expanded"
)
st.title("📄 Gemini‑Powered Financial Statement Extractor")
st.markdown(
    "Upload a **PDF financial statement** and extract structured transactions into Excel.\n"
    "If extraction fails, you can **download the raw response** to avoid wasting the run."
)


# ---------------- Helpers ----------------
def _mask_key(k: str) -> str:
    if not k or len(k) < 8:
        return "********"
    return f"{k[:4]}...{k[-4:]}"


def _load_available_api_keys() -> List[Tuple[str, str]]:
    """
    Returns list of tuples (label, key) for GOOGLE_API_KEY_1..3 found in st.secrets.
    """
    found = []
    for i in (1, 2, 3):
        name = f"GOOGLE_API_KEY_{i}"
        try:
            k = st.secrets[name]
            if k:
                found.append((name, k))
        except Exception:
            continue
    return found


def _try_extract_pdf_text_stats(file_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Optional local text stats via PyMuPDF; returns None if not available."""
    try:
        import fitz  # pymupdf
    except Exception:
        return None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text("text") or "" for page in doc]
        text = "\n".join(text_parts)
        pages = doc.page_count
        doc.close()
        char_count = len(text)
        word_count = len(text.split())
        est_tokens = max(int(char_count / 4), int(word_count * 0.75))
        return {"pages": pages, "char_count": char_count, "word_count": word_count, "estimated_tokens": est_tokens, "sample_text": text[:1200]}
    except Exception:
        return None


def _ensure_usage_log_in_state():
    if "usage_log" not in st.session_state:
        st.session_state["usage_log"] = []


def _append_usage_log(
    api_key_label: str,
    model_id: str,
    file_name: str,
    file_size_bytes: int,
    duration_sec: float,
    success: bool,
    error_msg: Optional[str],
    meta: Optional[Dict[str, Any]] = None
):
    _ensure_usage_log_in_state()
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "api_key_label": api_key_label,
        "model_id": model_id,
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "duration_sec": round(duration_sec, 3),
        "success": success,
        "error": error_msg or "",
    }
    if meta:
        entry.update(meta)
    st.session_state["usage_log"].append(entry)
    try:
        with open("usage_log.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state["usage_log"], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Configuration")

    # API Key selection
    st.markdown("### ✅ API Key")
    available_keys = _load_available_api_keys()
    api_key_label: str = "MANUAL"
    api_key: Optional[str] = None

    if available_keys:
        labels = [f"{name} ({_mask_key(k)})" for name, k in available_keys]
        default_idx = 0  # default to GOOGLE_API_KEY_1
        sel_idx = st.selectbox("Select API Key", options=range(len(labels)), index=default_idx,
                               format_func=lambda i: labels[i])
        api_key_label, api_key = available_keys[sel_idx]
        st.caption(f"Using API Key: **{_mask_key(api_key)}**")
    else:
        api_key = st.text_input(
            "Enter your Google Gemini API Key",
            type="password",
            key="api_key_input",
            help="No keys found in secrets. Paste a key here."
        )
        if api_key:
            st.caption(f"Using API Key: **{_mask_key(api_key)}**")
        else:
            st.warning("Please provide an API key to proceed.")

    # Model selection
    st.markdown("### ⚙️ Model")
    options = list(MODEL_CHOICES.keys())
    default_model_idx = options.index("Gemini 2.0 Flash") if "Gemini 2.0 Flash" in options else 0
    selected_model_name_display = st.selectbox(
        "Choose the Gemini Model",
        options=options,
        index=default_model_idx
    )
    MODEL_ID = MODEL_CHOICES[selected_model_name_display]
    st.info(f"Selected Model ID: `{MODEL_ID}`")

    # File upload
    st.markdown("### ⬆️ Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF Financial Statement", type=["pdf"])

    # Usage log download (kept in sidebar)
    st.markdown("---")
    st.subheader("📜 Usage Log")
    _ensure_usage_log_in_state()
    log_json = json.dumps(st.session_state["usage_log"], ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="⬇️ Download usage_log.json",
        data=log_json,
        file_name="usage_log.json",
        mime="application/json"
    )
    st.caption("Includes timestamp, key label, model, file, duration, success/error, counts, token estimate.")


# ---------------- Main ----------------
is_ready = (uploaded_file is not None) and (api_key is not None and api_key.strip() != "")

if uploaded_file:
    st.subheader("Uploaded File Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("File Name", uploaded_file.name)
    c2.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
    c3.metric("Model Selected", MODEL_ID)

    # Optional local PDF stats (tokens estimate)
    pdf_stats = None
    try:
        uploaded_file.seek(0)
        file_bytes_for_stats = uploaded_file.read()
        pdf_stats = _try_extract_pdf_text_stats(file_bytes_for_stats)
        if pdf_stats:
            with st.expander("📊 Local PDF Text Stats (approx.)", expanded=False):
                st.write({
                    "Pages": pdf_stats["pages"],
                    "Characters": pdf_stats["char_count"],
                    "Words": pdf_stats["word_count"],
                    "Estimated Tokens (rough)": pdf_stats["estimated_tokens"],
                })
                if pdf_stats.get("sample_text"):
                    st.code(pdf_stats["sample_text"][:600], language="text")
    except Exception:
        pass

    st.markdown("---")

    if not is_ready:
        st.error("Please complete the API Key and PDF upload steps in the sidebar.")
    else:
        start_button = st.button("🚀 Start Data Extraction", type="primary")

        if start_button:
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()

                t0 = time.time()
                with st.spinner(f"Extracting data using **{MODEL_ID}**..."):
                    extracted_data, raw_dump, error = extract_financial_statement(
                        api_key=api_key,
                        model_id=MODEL_ID,
                        pdf_file_path=uploaded_file.name,
                        file_bytes=file_bytes
                    )
                duration = time.time() - t0

                meta = {
                    "transactions": len(extracted_data.get("transaction_data", [])) if extracted_data else 0,
                    "debit_count": extracted_data.get("debit_count") if extracted_data else None,
                    "credit_count": extracted_data.get("credit_count") if extracted_data else None,
                    "alerts_count": len(extracted_data.get("alerts", [])) if (extracted_data and extracted_data.get("alerts")) else 0,
                    "estimated_tokens": (pdf_stats or {}).get("estimated_tokens") if 'pdf_stats' in locals() else None,
                }

                _append_usage_log(
                    api_key_label=(api_key_label if available_keys else "MANUAL"),
                    model_id=MODEL_ID,
                    file_name=uploaded_file.name,
                    file_size_bytes=uploaded_file.size,
                    duration_sec=duration,
                    success=(error is None),
                    error_msg=error,
                    meta=meta
                )

                if error:
                    st.error("Extraction failed.")
                    # Raw-response download to avoid wasting the run
                    st.subheader("Backup: Raw Gemini Response")
                    base_name = os.path.splitext(uploaded_file.name)[0]

                    if raw_dump:
                        raw_json_bytes = json.dumps(raw_dump, ensure_ascii=False, indent=2).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Raw Response (.json)",
                            data=raw_json_bytes,
                            file_name=f"{base_name}_gemini_raw.json",
                            mime="application/json"
                        )

                        raw_txt = ""
                        if raw_dump.get("response_text"):
                            raw_txt += "=== response_text ===\n" + str(raw_dump["response_text"]) + "\n\n"
                        if raw_dump.get("candidates_texts"):
                            for i, t in enumerate(raw_dump["candidates_texts"], 1):
                                raw_txt += f"=== candidate {i} ===\n{t}\n\n"

                        if raw_txt:
                            st.download_button(
                                label="⬇️ Download Raw Response (.txt)",
                                data=raw_txt.encode("utf-8"),
                                file_name=f"{base_name}_gemini_raw.txt",
                                mime="text/plain"
                            )
                        else:
                            st.info("No raw text available from the model response.")
                else:
                    st.success("✅ Done.")

                    # Alerts (unusual activity)
                    alerts = extracted_data.get("alerts", [])
                    if alerts:
                        with st.expander("⚠️ Unusual Activity / Transaction Alerts", expanded=True):
                            for a in alerts:
                                st.markdown(f"- {a}")

                    # Summary
                    st.subheader("Summary")
                    summary_cols = [
                        "institution_name", "account_holder_name", "statement_period",
                        "initial_balance", "closing_balance", "total_debit_amount",
                        "total_credit_amount", "debit_count", "credit_count"
                    ]
                    summary_data = {k.replace("_", " ").title(): extracted_data.get(k) for k in summary_cols}
                    summary_df = pd.DataFrame(summary_data.items(), columns=["Field", "Value"])
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    # Transactions Preview
                    st.subheader("Transactions Preview")
                    transaction_list = extracted_data.get("transaction_data", [])
                    if transaction_list:
                        df_preview = pd.DataFrame(transaction_list)
                        st.dataframe(df_preview.head(10), use_container_width=True)
                        st.info(f"Extracted **{len(transaction_list)}** transactions. Download the full file below.")

                        # Excel download
                        excel_buffer = json_to_excel_buffer(extracted_data)
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        st.download_button(
                            label="⬇️ Download Extracted Data (.xlsx)",
                            data=excel_buffer,
                            file_name=f"{base_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No transactions found.")

            except Exception:
                st.error("Unexpected application error.")
