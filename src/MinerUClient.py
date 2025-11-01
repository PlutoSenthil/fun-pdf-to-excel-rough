import fitz # PyMuPDF (alias)
from PIL import Image
from vllm import LLM
from mineru_vl_utils import MinerUClient

pdf_path = "your_document.pdf" # <-- **Replace with your PDF file path**
image = None
# --- 1. Load the PDF Page and Get Pixmap ---
try:
    with fitz.open(pdf_path) as doc:
        # Load the first page (index 0)
        page = doc.load_page(0)
        
        # Get the pixmap object. Use a high resolution (e.g., matrix=fitz.Matrix(2, 2))
        # to improve OCR quality for scanned documents.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # --- 2. Convert Pixmap to PIL Image ---
        # The 'mode' is 'RGB' and 'data' is the raw image bytes
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found.")
    exit()

# --- 3. Initialize vLLM and MinerUClient (as you provided) ---
# Note: This step assumes your environment has a compatible GPU and vLLM is installed correctly.
llm = LLM(model="opendatalab/MinerU2.5-2509-1.2B") 

client = MinerUClient(
    backend="vllm-engine", 
    vllm_llm=llm
)

# --- 4. Extract Data from the PIL Image ---
print("Extracting blocks using MinerU 2.5 with vLLM...")
extracted_blocks = client.two_step_extract(image)

# --- 5. Print Results ---
# The result will be a list of ContentBlock objects.
print(extracted_blocks)