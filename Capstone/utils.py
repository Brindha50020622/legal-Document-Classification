import torch
def classify_document(file_path, model, tokenizer):
    """Classify a single document into one of the categories."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoded_text = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    output = model(**encoded_text)
    prediction = torch.argmax(output.logits, dim=1).item()

    categories = ["Land", "Contract", "Judgment"]
    return categories[prediction]
