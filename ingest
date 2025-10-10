# ingest.py

import os
import chromadb
from sentence_transformers import SentenceTransformer
from backend import (
    extract_pdf,
    clean_text,
    sentence_based_chunking,
)

# --- CONFIGURATION ---
SOURCE_DIRECTORY = "source_folder"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CLIENT = chromadb.PersistentClient(path="chroma_db")

def main():
    """
    Processes all PDF files in the source directory and ingests them into ChromaDB.
    """
    print("Starting ingestion process...")
    
    pdf_files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in '{SOURCE_DIRECTORY}'.")
        return

    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(SOURCE_DIRECTORY, pdf_file)
            print(f"Processing '{pdf_file}'...")

            # Use 'rb' (read binary) mode for opening PDFs
            with open(file_path, "rb") as f:
                # We need a simple file-like object with a 'read' method and a 'name' attribute
                class FileObject:
                    def __init__(self, file_path, file_handle):
                        self.name = os.path.basename(file_path)
                        self.file_handle = file_handle
                    def read(self):
                        return self.file_handle.read()

                file_obj = FileObject(file_path, f)

                # 1. Extract, clean, and chunk text
                raw_text = extract_pdf(file_obj) # extract_pdf expects a file-like object
                cleaned_text = clean_text(raw_text)
                chunks = sentence_based_chunking(cleaned_text)

                if not chunks:
                    print(f"  -> Could not extract any text from '{pdf_file}'. Skipping.")
                    continue
                
                # 2. Create embeddings
                embeddings = EMBED_MODEL.encode(chunks, show_progress_bar=False)

                # 3. Store in ChromaDB
                doc_name = file_obj.name.replace('.pdf', '')
                collection = CLIENT.get_or_create_collection(name=doc_name)
                ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

                collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=ids
                )
                print(f"  -> Successfully ingested '{pdf_file}' into collection '{doc_name}'.")

        except Exception as e:
            print(f"  -> Failed to process '{pdf_file}'. Error: {e}")

    print("\nIngestion process finished.")

if __name__ == "__main__":
    main()