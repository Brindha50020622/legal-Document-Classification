import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the WordNet Lemmatizer and Porter Stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Define stop words
stop_words = set(stopwords.words('english'))

def clean_text(text, use_stemming=False):
    # Lowercasing
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Tokenize the text (split into words)
    words = text.split()
    
    # Remove stop words and apply lemmatization or stemming
    if use_stemming:
        words = [stemmer.stem(word) for word in words if word not in stop_words]
    else:
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join the words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def preprocess_and_save_text(extracted_text, output_file_path, use_stemming=False):
    # Preprocess the extracted text
    cleaned_text = clean_text(extracted_text, use_stemming)
    
    # Save the preprocessed text to a file
    with open(output_file_path, "w", encoding='utf-8') as preprocessed_file:
        preprocessed_file.write(cleaned_text)
    
    print(f"Preprocessed text saved to: {output_file_path}")

# Note: Remove the example usage if this script will be imported
