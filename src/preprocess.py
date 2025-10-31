import pdfplumber
import fitz  # PyMuPDF
import pytesseract
import easyocr
import cv2
from PIL import Image
import numpy as np
import os
import torch
# Auto-detect GPU availability
use_gpu = torch.cuda.is_available()
print(f"Using GPU: {use_gpu}")


def is_scanned_pdf(pdf_path):
    """Check if the PDF is scanned by inspecting the first page."""
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(gray)
        return len(text.strip()) < 10  # If very little text, likely scanned

def extract_text_pdfplumber(pdf_path):
    """Extract text from a text-based PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"\n--- Page {page_num + 1} ---")
            print(page.extract_text())

def extract_text_easyocr(pdf_path):
    """Extract text from scanned PDF using EasyOCR."""
    reader = easyocr.Reader(['en'], gpu=use_gpu)  # Add 'hi', 'ta', etc. for Indian languages
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = reader.readtext(np.array(img))
            print(f"\n--- Page {page_num + 1} ---")
            for (bbox, text, prob) in result:
                print(text)

def preprocess_pdf(pdf_path):
    if is_scanned_pdf(pdf_path):
        print("Detected scanned PDF. Using OCR...")
        extract_text_easyocr(pdf_path)
    else:
        print("Detected text-based PDF. Using pdfplumber...")
        extract_text_pdfplumber(pdf_path)

# Example usage
if __name__ == "__main__":
    pdf_file = "/home/simelabs/S-Pdf/project-root/"
    pdf_file = pdf_file + "Back_Dataset/Indian_Bank_Pic_Pg4.pdf"
    preprocess_pdf(pdf_file)