import fitz  # PyMuPDF
import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the BART tokenizer and model at the start
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text

def summarize_land_document(text):
    """Generate a summary for land documents based on specific details."""
    summary_lines = [
        "This is a land document issued by the Tamil Nadu Government, Department of Revenue."
    ]

    # Define patterns for extracting land document details
    details = {
        "District": r"District\s*:\s*(\w+)",
        "Circle": r"Circle\s*:\s*(\w+)",
        "Revenue Village": r"Revenue Village\s*:\s*(\w+)",
        "Patta No": r"Patta No\s*:\s*(\d+)",
        "Owner": r"Owners Name\s*\d+\.\s*([\w\s]+)",
        "Field Number": r"Field no subdivision\s*([\d\s-]+)",
        "Land Area": r"(\d+-\d+-\d+\.\d+)",
        "Reference Number": r"reference number\s*([\d/]+)",
        "Registration Date": r"(\d{2}-\d{2}-\d{4})"
    }

    # Find matches in the text and append details to the summary
    location = []
    for label, pattern in details.items():
        match = re.search(pattern, text)
        if match:
            if label in ["District", "Circle", "Revenue Village"]:
                location.append(f"{label}: {match.group(1)}")
            elif label == "Patta No":
                summary_lines.append(f"It references Patta No. {match.group(1)}.")
            elif label == "Owner":
                summary_lines.append(f"Listing {match.group(1).strip()} as the owner.")
            elif label == "Field Number":
                summary_lines.append(f"Specific land measurements include field number {match.group(1).strip()}.")
            elif label == "Reference Number":
                summary_lines.append(f"It confirms registration in the E-Registry with reference number {match.group(1)}.")
            elif label == "Registration Date":
                summary_lines.append(f"Registration date: {match.group(1)}.")

    # Append location details
    if location:
        summary_lines.insert(1, f"Location details: {', '.join(location)}.")

    # Return the final land document summary
    return "\n".join(summary_lines)

def summarize_text_with_bart(text, model, tokenizer, max_length=150, min_length=30, line_length=50):
    """Summarize text using the BART model with multiple line formatting, avoiding word splits."""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Split the summary without splitting words
    formatted_summary = ""
    words = summary.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > line_length:
            formatted_summary += line.strip() + "\n"
            line = ""
        line += word + " "
    formatted_summary += line.strip()
    
    return formatted_summary

def generate_summary(file_path):
    """Generate a summary for a PDF file, using either specific land document summarization or BART model summarization."""
    pdf_text = extract_text_from_pdf(file_path)
    
    # Check for keywords specific to a land document
    if "Patta No" in pdf_text:  # Assuming "Patta No" is unique to land documents
        return summarize_land_document(pdf_text)
    else:
        # For other document types, use BART model for summarization
        return summarize_text_with_bart(pdf_text, model, tokenizer)
