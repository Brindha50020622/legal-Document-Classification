import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import classify_document

def main(preprocessed_file_path):
    tokenizer = AutoTokenizer.from_pretrained("saved_model/")
    model = AutoModelForSequenceClassification.from_pretrained("saved_model/")

    # Set the model to evaluation mode
    model.eval()

    # Use classify_document function to classify a document
    #print(f"Classification of '{preprocessed_file_path}': {classify_document(preprocessed_file_path, model, tokenizer)}")
    print("The given Document is:", {classify_document(preprocessed_file_path, model, tokenizer)})

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_test.py <path_to_preprocessed_file>")
    else:
        main(sys.argv[1])
