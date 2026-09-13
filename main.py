from pathlib import Path
import json
from sentence_transformers import SentenceTransformer


# Locate the original document collection.
project_dir = Path(__file__).resolve().parent
collection_dir = (
    project_dir / "data" / "raw" / "AI_Technical_Documentation"
)


# Load the document inventory from the JSONL manifest.
manifest_path = collection_dir / "manifest.jsonl"
documents = []

with manifest_path.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            documents.append(json.loads(line))


# Attach each document's text to its metadata.
for document in documents:
    document_path = collection_dir / document["local_path"]
    document["text"] = document_path.read_text(encoding="utf-8")


# Confirm how many documents were loaded.
print(f"Loaded {len(documents)} documents.")

# Split documents into overlapping passages and retain their sources.
chunks = []
chunk_size = 150
overlap = 30

for document in documents:
    words = document["text"].split()

    for start in range(0, len(words), chunk_size - overlap):
        chunk_words = words[start:start + chunk_size]

        chunks.append({
            "text": " ".join(chunk_words),
            "title": document["title"],
            "source_url": document["source_url"],
            "document_id": document["document_id"],
        })

        if start + chunk_size >= len(words):
            break

print(f"Created {len(chunks)} chunks.")
print("First chunk:", chunks[0]["text"])

# Load BGE Small on CPU and embed three passages as a first check.
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cpu",
)

sample_texts = [chunk["text"] for chunk in chunks[:3]]

sample_embeddings = embedding_model.encode(
    sample_texts,
    normalize_embeddings=True,
)

print("Embedding shape:", sample_embeddings.shape)

# Inspect the first ten values in the first passage's embedding.
print("First embedding values:", sample_embeddings[0][:10])