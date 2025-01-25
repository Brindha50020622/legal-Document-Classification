import os
import pytesseract as tess
from pdf2image import convert_from_path
from summary import generate_summary  # Import the summarization function
from preprocessing import preprocess_and_save_text  # Import the updated preprocessing function
import subprocess
import streamlit as st

# Set the Tesseract command path
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def read_pdf(file):
    pages = []

    try:
        # Convert the PDF file to a list of PIL images
        images = convert_from_path(file)

        # Extract text from each image
        for i, image in enumerate(images):
            # Extract text from each image using pytesseract
            text = tess.image_to_string(image)
            pages.append(text)  # Append extracted text to pages list

    except Exception as e:
        st.error(f"Error processing the PDF: {str(e)}")
        return ""

    # Return the combined extracted text
    return "\n".join(pages)

def main():
    # Streamlit app layout
    st.title("PDF Text Extraction and Processing")

    # Step 1: Upload PDF file
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded PDF temporarily
        pdf_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Step 2: Read and extract text from the PDF
        extracted_text = read_pdf(pdf_path)

        if extracted_text:
            # Step 3: Preprocess the extracted text and save to a file
            preprocessed_file_path = os.path.splitext(pdf_path)[0] + "_preprocessed.txt"
            preprocess_and_save_text(extracted_text, preprocessed_file_path)

            st.success(f"Preprocessed text saved to: {preprocessed_file_path}")

            # Step 4: Ask user for classification or summarization
            choice = st.radio("Choose an option:", ("Classify the text", "Summarize the text"))

            if choice == "Classify the text":
                # Get the absolute path of the current script's directory
                current_dir = os.path.dirname(os.path.abspath(__file__))

                # Build the absolute path to model_test.py
                model_test_path = os.path.join(current_dir, "model_test.py")

                # Run model_test.py with the preprocessed file path
                result = subprocess.run(
                    ["python", model_test_path, preprocessed_file_path],
                    capture_output=True,
                    text=True
                )
                st.success("Classification completed.")
                if result.stdout:
                    st.write(f"**Classification Result:** {result.stdout}")
                if result.stderr:
                    st.error(f"Error during classification: {result.stderr}")

            elif choice == "Summarize the text":
                # Summarize the original extracted text using the generate_summary function
                summary = generate_summary(pdf_path)  # Use the original PDF path for summarization
                st.write(f"Summary:\n{summary}")

if __name__ == "__main__":
    main()
