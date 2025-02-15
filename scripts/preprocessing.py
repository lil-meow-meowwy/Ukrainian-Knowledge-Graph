import pandas as pd

OUTPUT_CSV = "extracted_entities.csv"
PREPROCESSED_CSV = "processed_data.csv"
FIRST_CHUNK = True
CHUNK_SIZE = 10000  

for chunk in pd.read_csv(OUTPUT_CSV, chunksize=CHUNK_SIZE):
    chunk.drop_duplicates(inplace=True)
    chunk.dropna(subset=["text"], inplace=True)

    mode = "w" if FIRST_CHUNK else "a"  # Write first time, then append
    header = FIRST_CHUNK  # Write header only once

    chunk.to_csv(PREPROCESSED_CSV, mode=mode, header=header, index=False)
    FIRST_CHUNK = False  # After first write, switch to append mode
    print(f"[INFO] Saved {len(chunk)} rows")