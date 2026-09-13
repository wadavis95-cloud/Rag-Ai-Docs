# Import tools for working with paths and reading JSON records.
from pathlib import Path
import json


# Find the folder containing this script and locate our raw data.
project_dir = Path(__file__).resolve().parent
raw_dir = project_dir / "data" / "raw"


# Point to the extracted collection and list its immediate contents.
collection_dir = raw_dir / "AI_Technical_Documentation"

for item in collection_dir.iterdir():
    print(item.name)


# Read the first manifest record and convert it into a dictionary.
# Each line in this JSONL file describes one document.
manifest_path = collection_dir / "manifest.jsonl"

with manifest_path.open("r", encoding="utf-8") as file:
    first_document = json.loads(file.readline())


# Inspect the document's identity, source, and recorded quality flags.
# An empty flags list does not guarantee that the document is clean.
print("Title:", first_document["title"])
print("Project:", first_document["project"])
print("Source:", first_document["source_url"])
print("Quality flags:", first_document["quality_flags"])


# Use the path from the manifest to open the original document.
# Reading the file leaves its contents unchanged.
document_path = collection_dir / first_document["local_path"]

with document_path.open("r", encoding="utf-8") as file:
    document_text = file.read()


# Preview the first 1,500 characters so we can inspect the formatting
# without printing the entire document in the terminal.
print(document_text[:1500])

#Find the first embedded reStructuredText block in the Documen
marker = "```{eval-rst}"
start = document_text.find(marker)

#Preview the block only if the marker exists. 
#find() returns -1 when it cannot find the requested text.
if start != -1:
    print(document_text[start:start + 1200])
else:
    print("No eval-rst block found.")

# Preview the first converted label alongside its original surrounding lines.
lines = document_text.splitlines()

for index, line in enumerate(lines):
    stripped_line = line.strip()

    if stripped_line.startswith(".. tab::"):
        label = stripped_line.removeprefix(".. tab::").strip()

        print("### " + label)
        print("\n".join(lines[index + 1:index + 6]))
        break

