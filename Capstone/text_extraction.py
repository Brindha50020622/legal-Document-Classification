import fitz  # PyMuPDF

def read_pdf(file):
    """Extracts text directly from a PDF without converting to images."""
    try:
        with fitz.open(file) as doc:
            text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()  # Extract text from each page

            return text  # Return the full extracted text
    except Exception as e:
        raise Exception(f"Error processing {file}: {str(e)}")
