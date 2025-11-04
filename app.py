import streamlit as st
from gemini import GeminiExtractor

st.set_page_config(page_title="Bank Statement → Excel", page_icon="📄")

st.title("📄 Bank Statement → Excel (Gemini)")
st.caption("Minimal demo: upload one PDF bank statement, extract transactions as Excel.")

# Read secrets
api_key = st.secrets.get("GOOGLE_API_KEY_1", "")
model_id = st.secrets.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

if not api_key:
    st.error("Please set GOOGLE_API_KEY_1 in Streamlit secrets.")
    st.stop()

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded and st.button("Extract → Download Excel"):
    with st.spinner("Processing with Gemini..."):
        try:
            extractor = GeminiExtractor(api_key=api_key, model_id=model_id)
            excel_io = extractor.process_pdf_to_excel(uploaded.read())
            st.success("Done!")
            st.download_button(
                "⬇️ Download Excel",
                data=excel_io.getvalue(),
                file_name="bank_statement.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Failed to process file: {e}")