import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

def load_texts_from_folder(folder_path, label):
    """Load texts and labels from the specified folder."""
    texts, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

def classify_document(file_path, model, tokenizer):
    """Classify a single document into one of the four categories."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoded_text = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    output = model(**encoded_text)
    prediction = torch.argmax(output.logits, dim=1).item()

    categories = ["Land", "Contract", "Judgment"]
    return categories[prediction]

def main():
    # 1. Load the Legal BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased", num_labels=3 
    )

    land_texts, land_labels = load_texts_from_folder("data/land", label=0)
    contract_texts, contract_labels = load_texts_from_folder("data/Contracts", label=1)
    judgment_texts, judgment_labels = load_texts_from_folder("data\\judgements\\preprocessed", label=2)
 

    print(f"Number of Land documents: {len(land_labels)}")
    print(f"Number of Contract documents: {len(contract_labels)}")
    print(f"Number of Judgment documents: {len(judgment_labels)}")
   

    
    texts = land_texts + contract_texts + judgment_texts
    labels = land_labels + contract_labels + judgment_labels

    
    encoded_data = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded_data["input_ids"]
    attention_mask = encoded_data["attention_mask"]

    # 5. Split the data into train and test sets
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=42
    )
    train_mask, test_mask = train_test_split(
        attention_mask, test_size=0.2, random_state=42
    )

    # 6. Create a DataLoader for training
    train_data = TensorDataset(train_ids, train_mask, torch.tensor(train_labels))
    train_loader = DataLoader(train_data, batch_size=8)

    # 7. Train the model 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(5):  
        for batch in train_loader:
            b_input_ids, b_attention_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=test_ids, attention_mask=test_mask)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()

    # Print classification report
    print(classification_report(
        test_labels, predictions,
        target_names=["Land", "Contract", "Judgment"],
        labels=[0, 1, 2]
    ))
    print(f"Predictions: {predictions}")
    print(f"True Labels: {test_labels}")
    model.save_pretrained("saved_model/")
    tokenizer.save_pretrained("saved_model/")
    print("Model saved.")

    # 9. Classify a new document
    example_file = "data/contract_100.txt"
    print(f"Classification of '{example_file}': {classify_document(example_file, model, tokenizer)}")


if __name__ == "__main__":
    main()
