"""CLI tool to batch-ingest documents into the Aura RAG vector store."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_engine import ingest_documents, DOCUMENTS_DIR

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Aura RAG")
    parser.add_argument("paths", nargs="+", help="File or directory paths to ingest")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    args = parser.parse_args()

    file_paths = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(path)
            else:
                print(f"Skipping unsupported file: {path} ({path.suffix})")
        elif path.is_dir():
            pattern = "**/*" if args.recursive else "*"
            for f in path.glob(pattern):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    file_paths.append(f)
        else:
            print(f"Path not found: {path}")

    if not file_paths:
        print("No ingestible files found.")
        sys.exit(1)

    print(f"Found {len(file_paths)} file(s) to ingest...")

    # Use Streamlit's UploadedFile-like wrapper
    class SimpleUploadedFile:
        def __init__(self, path: Path):
            self.name = path.name
            self.size = path.stat().st_size
            self._path = path

        def getbuffer(self):
            return self._path.read_bytes()

        def read(self):
            return self._path.read_bytes()

    uploaded = [SimpleUploadedFile(f) for f in file_paths]
    count = ingest_documents(uploaded)
    print(f"Ingested {count} chunks into the vector store.")


if __name__ == "__main__":
    main()
