# Rag-Ai-Docs

A simple local retrieval-augmented generation (RAG) capstone for answering questions about AI technical documentation.

**Milestone completed:** documents are loaded, chunked, embedded, retrieved, and passed to a language model that generates an answer. A focused answer was manually checked against its supporting evidence. Broader questions exposed retrieval and answer-grounding limitations.

This is a working learning baseline. It has not been evaluated with a formal benchmark or established as reliable across questions.

## How it works

```text
150 source documents → overlapping chunks → embeddings
Question → query embedding → top 3 passages → Qwen → answer
```

The script prints the answer and the retrieved passage titles, similarity scores, source URLs, and text so they can be inspected together. Source labels are requested in the answer, but citation compliance is not enforced by code.

| Component | Current setting |
|---|---|
| Collection | All 150 records in the supplied manifest |
| Chunking | 150 whitespace-separated words, 30-word overlap |
| Embeddings | `BAAI/bge-small-en-v1.5`, CPU, batch size 16 |
| Retrieval | Normalized embeddings; dot product equals cosine similarity; top 3 passages |
| Query encoding | Search instruction prefixed to the question |
| Answer model | `Qwen/Qwen3-0.6B`, CPU, float32 |
| Generation | Thinking disabled in chat template; sampling disabled; up to 256 new tokens |
| PyTorch threads | 8 |
| Answer rules | Use supplied evidence, be specific, cite sources, preserve commands/version requirements, acknowledge missing evidence |

There is one question per run, set in `main.py`. Each run rebuilds the passage embeddings in memory. No vector database, persistent index, web service, or orchestration framework is required by this implementation.

## Run locally

The successful run was on a Windows desktop. The current local environment was inspected when documenting this milestone: Python 3.14.7, PyTorch 2.14.0, Transformers 5.17.0, Sentence Transformers 6.0.1, and NumPy 2.5.1. These package versions are captured in `requirements.txt`; this records the environment at documentation time, not a verified package inventory for every earlier experiment.

The commands below use PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m zipfile -e docs/AI_Technical_Documentation_150.zip data/raw
.\.venv\Scripts\python.exe main.py
```

Extract the archive once if the collection is not already present. Its top-level folder must resolve to:

```text
data/raw/AI_Technical_Documentation/manifest.jsonl
data/raw/AI_Technical_Documentation/documents/...
```

The first model load downloads weights from Hugging Face; later runs can use the local model cache. These fresh-environment installation commands have not been independently retested. The model IDs are specified in code without revision pins.

Change `question` in `main.py` to try another question. The script should report `Loaded 150 documents.` before embedding the chunks. Only the top three passages are sent to the answer model.

## What the runs showed

- **Supported answer:** a question about custom loss functions correctly identified the required return value and optional model-card metadata described by the evidence.
- **Broader explanation:** a question explicitly mentioning embeddings retrieved relevant material and produced a broadly supported explanation.
- **Everyday wording:** asking how an app can find information when users choose different words retrieved tokenization and custom-module details, then produced an unsupported recommendation involving a `task` argument.
- **Instruction issue:** missing citations were initially over-attributed to the model. One system-prompt revision had left an incomplete citation instruction. The latest rules explicitly request specific answers and evidence citations.
- **Runtime issue:** earlier Codespaces runs ended with SIGTERM. A later desktop run succeeded, but the termination cause and the effect of reducing CPU threads were not established.

A similarity score such as `0.64` is not 64% accuracy or a confidence estimate. The top-ranked passages can still be insufficient to answer a question.

## Scope and lessons

The three project rules are **simplicity, fidelity, and cost effectiveness**: keep the pipeline understandable, preserve and check source meaning, and spend time or compute on changes with a clear learning purpose. A completed end-to-end milestone does not imply that every answer is correct.

The corpus contains original Markdown/reStructuredText snapshots from Hugging Face Transformers, Sentence Transformers, and Accelerate. These are pinned development commits, not stable-release API guarantees. The baseline chunks raw text with `split()`, so source formatting, code structure, and directive context may be lost. Source metadata, licenses, and notices are retained in the collection archive.

See the [build journal](docs/build-journal.md) for experiments, errors, corrections, and unresolved questions, and the [first inspection notes](docs/session-notes.md) for the initial document investigation. Future work may explore evaluation, retrieval changes, or more complex system design; those are separate learning objectives rather than requirements for this completed baseline.
