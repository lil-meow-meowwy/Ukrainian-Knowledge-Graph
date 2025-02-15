import spacy
import os
import json
import pandas as pd
import time

OUTPUT_CSV = "extracted_entities.csv"
DIRECTORY = '/wiki_dump/extracted_text' #directory with dump
BATCH_SIZE = 100  # Save progress every 100 articles

# Load spaCy's multilingual model (supports Ukrainian)
print("[INFO] Loading spaCy model...")
nlp = spacy.load("xx_ent_wiki_sm")  # Faster Ukrainian-only model

nlp.enable_pipe("ner")  # Ensure NER is enabled

if spacy.prefer_gpu():
    print("[INFO] Using GPU acceleration for spaCy.")
else:
    print("[INFO] Running on CPU.")

# Function to extract named entities from text using spaCy
def extract_entities_from_text(text):
    """Extract named entities using spaCy."""
    try:
        doc = nlp(text)
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    except Exception as e:
        print(f"[ERROR] NER processing failed: {e}")
        return []

# Process and save entities in chunks
def process_articles(directory):
    """Loads, processes, and saves articles in chunks to reduce memory usage."""
    article_count = 0
    extracted_data = []
    start_time = time.time()

    print(f"[INFO] Starting entity extraction from {directory}...")

    # Walk through all files in the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"[INFO] Processing file: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        article = json.loads(line)
                        title = article.get('title', 'Unknown Title')
                        text = article.get('text', '')

                        if text:
                            entities = extract_entities_from_text(text)
                            extracted_data.append({'title': title, 'text': text, 'entities': entities})
                            article_count += 1
                        else:
                            print(f"[WARNING] Skipping empty article: {title}")

                    except json.JSONDecodeError:
                        print(f"[WARNING] Invalid JSON in {file_path}. Skipping.")

                    # Save every BATCH_SIZE articles
                    if article_count % BATCH_SIZE == 0 and article_count > 0:
                        print(f"[INFO] Processed {article_count} articles. Saving to CSV...")
                        save_to_csv(extracted_data)
                        extracted_data = []  # Clear memory

    # Save any remaining data
    if extracted_data:
        print(f"[INFO] Final batch of {len(extracted_data)} articles. Saving to CSV.")
        save_to_csv(extracted_data)

    elapsed_time = time.time() - start_time
    print(f"[INFO] Finished processing {article_count} articles in {elapsed_time:.2f} seconds.")

# Save extracted entities to CSV
def save_to_csv(data):
    """Appends extracted entity data to a CSV file."""
    try:
        df = pd.DataFrame([
            {'title': item['title'], 'text': item['text'], 'entity': ent['text'], 'label': ent['label']}
            for item in data for ent in item['entities']
        ])
        
        df.to_csv(OUTPUT_CSV, mode='a', index=False, encoding='utf-8', header=not os.path.exists(OUTPUT_CSV))
        print(f"[INFO] Saved {len(data)} articles to {OUTPUT_CSV}")

    except Exception as e:
        print(f"[ERROR] Failed to save to CSV: {e}")

process_articles(DIRECTORY)