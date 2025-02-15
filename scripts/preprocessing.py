import pandas as pd
import re
import unicodedata
from tqdm import tqdm  # For progress bar
from multiprocessing import Pool, cpu_count
import os

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
    return True

# Function to process a single chunk
def process_chunk(chunk):
    """Process a single chunk of data."""
    chunk['text'] = chunk['text'].apply(clean_text)
    chunk['entity'] = chunk['entity'].apply(clean_text)
    
    # Filter entities
    chunk = chunk[chunk.apply(lambda x: filter_entity(x['entity'], x['label']), axis=1)]
    
    # Deduplicate entities within the same article
    chunk = chunk.drop_duplicates(subset=['title', 'entity', 'label'])
    
    return chunk

# Function to preprocess the data in parallel
def preprocess_data(input_csv, output_csv):
    """Preprocess the extracted entities in parallel."""
    print(f"[INFO] Loading data from {input_csv}...")
    chunksize = 10**6  # Process in chunks to handle large files
    total_rows = sum(1 for _ in open(input_csv, 'r', encoding='utf-8')) - 1  # Get total rows for progress bar
    num_processes = cpu_count()  # Use all available CPU cores

    # Initialize the output file with headers
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['title', 'text', 'entity', 'label']).to_csv(output_csv, index=False, encoding='utf-8')

    # Process chunks in parallel
    with Pool(num_processes) as pool:
        for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize, encoding='utf-8'), total=total_rows//chunksize + 1):
            processed_chunks = pool.map(process_chunk, [chunk])
            
            # Append processed chunks to the output file
            for processed_chunk in processed_chunks:
                processed_chunk.to_csv(output_csv, mode='a', index=False, encoding='utf-8', header=False)

    print("[INFO] Preprocessing completed.")

# Run the preprocessing
preprocess_data(INPUT_CSV, OUTPUT_CSV)