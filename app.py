import streamlit as st
from gemini import GeminiExtractor

st.set_page_config(page_title="Bank Statement Extractor", page_icon="📄")

st.title("📄 Bank Statement → Excel")
st.write("Upload a PDF bank statement and download the extracted transactions as Excel.")

# Get API key from Streamlit secrets
api_key = st.secrets.get("GOOGLE_API_KEY_1", "")
model_id = st.secrets.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

if not api_key:
    st.error("Please set GOOGLE_API_KEY_1 in Streamlit secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.button("Extract and Download"):
        with st.spinner("Processing..."):
            try:
                extractor = GeminiExtractor(api_key=api_key, model_id=model_id)
                excel_file = extractor.process_pdf(uploaded_file.read())
                st.success("Extraction complete!")
                st.download_button(
                    label="Download Excel",
                    data=excel_file.getvalue(),
                    file_name="bank_statement.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Failed to process file: {e}")