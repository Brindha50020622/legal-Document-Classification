import os
import streamlit as st
import subprocess
from text_extraction import read_pdf  # Import the extraction function
from summary import generate_summary  # Import summarization function
from preprocessing import preprocess_and_save_text  # Import preprocessing function

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
        try:
            extracted_text = read_pdf(pdf_path)  # Call the function from text_extraction.py

            if extracted_text:
                # Step 3: Preprocess and save extracted text
                preprocessed_file_path = os.path.splitext(pdf_path)[0] + "_preprocessed.txt"
                preprocess_and_save_text(extracted_text, preprocessed_file_path)

                st.success(f"Preprocessed text saved to: {preprocessed_file_path}")

                # Step 4: Ask user to choose between classification or summarization
                choice = st.radio("Choose an option:", ("Classify the text", "Summarize the text"))

                if choice == "Classify the text":
                    # Step 5: Run the classification script and capture output
                    result = subprocess.run(
                        ["python", "model_test.py", preprocessed_file_path],
                        capture_output=True, text=True
                    )

                    # Display the classification output in Streamlit
                    if result.returncode == 0:
                        st.success("Classification completed successfully.")
                        st.write("**Classification Output:**")
                        st.code(result.stdout)  # Display the output as code block
                    else:
                        st.error(f"Error during classification:\n{result.stderr}")

                elif choice == "Summarize the text":
                    # Generate summary of the original text
                    summary = generate_summary(pdf_path)  # Use extracted text
                    st.write(f"Summary:\n{summary}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
