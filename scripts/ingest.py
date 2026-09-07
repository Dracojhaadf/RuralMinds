"""
Bulk document ingestion utility.
Ingests PDF files from the source directory into ChromaDB and registers them in SQLite.
"""
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import SOURCE_FOLDER, UPLOADED_PDFS_FOLDER
from services.document_service import process_and_save_pdf

def main():
    print("=" * 60)
    print("Starting Ingestion Process...")
    print("=" * 60)
    
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Directory '{SOURCE_FOLDER}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in '{SOURCE_FOLDER}'.")
        return

    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(SOURCE_FOLDER, pdf_file)
            print(f"\nProcessing '{pdf_file}'...")

            with open(file_path, "rb") as f:
                success, msg = process_and_save_pdf(f, original_filename=pdf_file)
                if success:
                    print(f"  ✅ {msg}")
                else:
                    print(f"  ❌ {msg}")

        except Exception as e:
            print(f"  ❌ Error processing '{pdf_file}': {str(e)}")

    print("\n" + "=" * 60)
    print("Ingestion Process Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
