import pandas as pd
import re
import unicodedata
from tqdm import tqdm  # For progress bar

# Input and output file paths
INPUT_CSV = "data/extracted_entities.csv"
OUTPUT_CSV = "data/preprocessed_entities.csv"

# Define stopwords
STOPWORDS = set(["і", "та", "у", "в", "на", "з", "зі", "для", "що", "як", "але", "це"])

# Function to clean and normalize text
def clean_text(text):
    """Remove special characters, extra spaces, and normalize text."""
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode characters (e.g., remove diacritics)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip().lower()  # Convert to lowercase and strip leading/trailing spaces
    
    return text

# Function to filter entities
def filter_entity(entity, label):
    """Filter out irrelevant entities or stopwords."""
    if not entity or len(entity) < 2:  # Remove very short entities
        return False
    if entity in STOPWORDS:  # Remove stopwords
        return False
    if label in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"]:  # Remove irrelevant labels
        return False
    return True

# Function to preprocess the data
def preprocess_data(input_csv, output_csv):
    """Preprocess the extracted entities."""
    print(f"[INFO] Loading data from {input_csv}...")
    chunksize = 10**6  # Process in chunks to handle large files
    processed_data = []

    # Read the CSV file in chunks
    for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize, encoding='utf-8')):
        chunk['text'] = chunk['text'].apply(clean_text)
        chunk['entity'] = chunk['entity'].apply(clean_text)
        
        # Filter entities
        chunk = chunk[chunk.apply(lambda x: filter_entity(x['entity'], x['label']), axis=1)]
        
        # Deduplicate entities within the same article
        chunk = chunk.drop_duplicates(subset=['title', 'entity', 'label'])
        
        processed_data.append(chunk)

    # Concatenate all processed chunks
    print("[INFO] Concatenating processed chunks...")
    processed_df = pd.concat(processed_data, ignore_index=True)
    
    # Save the preprocessed data to a new CSV file
    print(f"[INFO] Saving preprocessed data to {output_csv}...")
    processed_df.to_csv(output_csv, index=False, encoding='utf-8')
    print("[INFO] Preprocessing completed.")

# Run the preprocessing
preprocess_data(INPUT_CSV, OUTPUT_CSV)