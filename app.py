# app.py
import os
import io
import zipfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from gemini import GeminiExtractor

load_dotenv()  # Loads .env if present


def get_api_key() -> str:
    # Priority: Streamlit secrets -> env -> UI input
    key = st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = os.getenv("GOOGLE_API_KEY", "")
    return key


def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


st.set_page_config(page_title=" Statement → Excel", page_icon="📄", layout="centered")

st.title("📄→📊  Statement Extractor")
st.caption("Upload  statement PDFs. Each PDF will be extracted into a structured Excel file.")

with st.expander("⚠️ Security & Notes", expanded=False):
    st.markdown("- For folder uploads, please **zip the folder** and upload the ZIP.")

# --- Sidebar: Config ---
with st.sidebar:
    st.header("Configuration")
    default_model = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
    model_id = st.text_input("Model ID", value=default_model, help="e.g., gemini-2.5-flash")

    api_key_default = get_api_key()
    api_key = st.text_input("Google API Key", value=api_key_default, type="password")
    st.markdown("Set `GOOGLE_API_KEY` in environment variables or Streamlit secrets for convenience.")

    custom_prompt = st.text_area(
        "Extraction Prompt (optional, advanced)",
        value="",
        placeholder="Leave blank to use the default prompt tuned for  statements.",
        height=120,
    )

st.markdown("---")

# --- File Uploader ---
files = st.file_uploader(
    "Upload one or more **PDFs** or a **ZIP** containing PDFs",
    type=["pdf", "zip"],
    accept_multiple_files=True,
    help="Multiple PDFs supported. For folders, zip the folder before uploading."
)

process_clicked = st.button("🚀 Process Files")

if process_clicked:
    if not api_key:
        st.error("Please provide a valid Google API key.")
        st.stop()

    if not files:
        st.warning("Please upload at least one PDF or a ZIP file.")
        st.stop()

    # Initialize extractor
    extractor = GeminiExtractor(
        api_key=api_key,
        model_id=model_id or "gemini-2.5-flash",
        prompt=custom_prompt.strip() or None  # None will default inside class
    )

    # Collect PDFs from uploads (support PDFs and/or ZIPs containing PDFs)
    pdf_blobs = []

    for f in files:
        if f.type == "application/pdf" or (f.name.lower().endswith(".pdf")):
            pdf_blobs.append((f.name, f.read()))
        elif f.type in ("application/zip", "application/x-zip-compressed") or f.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".pdf") and not name.endswith("/"):
                            pdf_blobs.append((name.split("/")[-1], zf.read(name)))
            except zipfile.BadZipFile:
                st.error(f"'{f.name}' appears to be a corrupted ZIP.")
        else:
            st.warning(f"Unsupported file type: {f.name}")

    if not pdf_blobs:
        st.error("No PDFs found to process.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    output_zip = io.BytesIO()
    zip_buffer = zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED)

    results_container = st.container()

    for i, (fname, fbytes) in enumerate(pdf_blobs, start=1):
        status.info(f"Processing {fname} ({i}/{len(pdf_blobs)}) ...")
        try:
            extracted_json, excel_bytes = extractor.process_pdf_to_excel(fbytes, fname)
            base = sanitize_filename(fname.rsplit(".", 1)[0])
            xlsx_name = f"{base}.xlsx"

            # Show quick preview of transactions
            with results_container.expander(f"✅ {fname} → {xlsx_name}", expanded=False):
                st.json(extracted_json)
                st.download_button(
                    label=f"⬇️ Download {xlsx_name}",
                    data=excel_bytes.getvalue(),
                    file_name=xlsx_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Add to zip
            zip_buffer.writestr(xlsx_name, excel_bytes.getvalue())

        except Exception as e:
            with results_container.expander(f"❌ {fname} - Error", expanded=True):
                st.error(f"Failed to process {fname}.\n\n**Details:** {e}")

        progress.progress(i / len(pdf_blobs))

    # Close and offer ZIP
    zip_buffer.close()
    output_zip.seek(0)
    status.success("All done!")

    st.download_button(
        label="⬇️ Download All as ZIP",
        data=output_zip.getvalue(),
        file_name="statements_excel.zip",
        mime="application/zip"
    )