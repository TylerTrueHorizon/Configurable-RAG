# Keyword-Based Search

This project provides a simple command-line tool for keyword search across text documents using the Okapi BM25 ranking algorithm.

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare a folder containing your text documents (e.g. `.txt`, `.md`).

## Usage

Run the search script with your document directory and query:

```bash
python main.py --docs /path/to/documents --query "search terms" --top_n 5 --chunk_size 100
```

- `--docs` – path to the folder with documents.
- `--query` – search keywords.
- `--top_n` – number of results to display (default: 5).
- `--chunk_size` – number of words per chunk (default: 100).

The script returns the most relevant chunks and their document names.
