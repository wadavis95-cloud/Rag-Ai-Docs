# Build journal: local RAG baseline

Milestone recorded: September 14, 2026.

## Objective and outcome

The initial question was: **does this simple RAG system work end to end?** The answer is yes. The pipeline loads source documents, splits them into chunks, embeds them, retrieves passages for a question, and generates an answer from that context. At least one focused answer was manually checked and matched the supplied evidence.

Subsequent questions explored answer quality. They revealed limitations, but did not change the fact that the initial integration milestone was complete. These observations are informal reviews of individual runs, not a formal evaluation dataset, aggregate quality score, or repeatability benchmark.

## From document inspection to a working run

The collection contains 150 technical-document source files: 85 from Transformers, 35 from Sentence Transformers, and 30 from Accelerate. Manifest records retain document IDs and source URLs pinned to Git commits. The original [inspection session notes](session-notes.md) record the first manifest/document read and a small preview of converting documentation tab labels into a heading while retaining the installation command.

Early setup problems included a missing raw-data folder, folder-name capitalization mismatch, incorrect search text, loop indentation, and a VS Code Run-button issue. Correcting the local setup and running `python main.py` allowed the inspection work to proceed. There was also a missing-document/manifest-path problem during development; the collection is now present at the path used by the script.

The cleaning preview did not become a full ingestion-cleaning stage. The current baseline reads the original files and chunks whitespace-separated words. That is simple, but can split code, flatten structure, and retain documentation markup or unexpanded directives. Empty quality flags alone do not prove that a source is ready for use as answer evidence.

The first pipeline used ten manifest records. The loading limit was then removed to use all 150. Chunk size, overlap, embedding model, and top-three retrieval remained the same. Expanding the collection increases searchable coverage; it does not guarantee that the right passages will rank highest.

## Runtime investigation

Earlier Codespaces runs terminated with SIGTERM. The local termination trace records that signal, but does not establish the underlying reason. It does not prove an out-of-memory event, a model failure, or a CPU-thread problem.

PyTorch CPU threads were reduced to one during investigation. A later home-desktop run completed after the environment and prompt had changed. Those simultaneous changes prevent attributing success to a single cause. The latest script uses eight threads; no controlled timing comparison establishes whether eight is faster than one or four for this workload.

## Questions and observed results

| Experiment | Observed result | What it establishes |
|---|---|---|
| Ask for the steps to turn documents into a question-answering system | Retrieved custom-loss and custom-model details. The answer incorrectly presented them as the requested steps. Two loss passages overlapped. | The selected evidence did not answer the question, and the answer made an unsupported connection. |
| Ask what a custom loss must return and which additions support model cards | Answer identified a single loss value or dictionary of loss components, plus optional `get_config_dict` and `citation`. | One focused answer matched the supplied passage. |
| Ask how embeddings help find information relevant to a question | Retrieved material on task-dependent similarity, query/document embeddings, and retrieval scoring. Produced a broadly supported explanation without citation labels. | Useful synthesis occurred on this question. It was not proof of general reliability. |
| Ask in everyday wording: “How can I help my app find the right information even when someone uses different words than the documents?” | Retrieved token classification and custom-module `task`-argument examples. Answer claimed the argument enabled the desired matching behavior. | These top-three passages did not support the proposed solution. |
| Keep the everyday question and temporarily supply the earlier embeddings passages manually | Answer shifted toward multi-vector encoding and query/document routes, but remained overly specific and asserted an improvement not established by those passages. | The answer changed with the evidence. This did not cleanly isolate the model from evidence-quality problems. |

The earlier successful technical wording and unsuccessful everyday wording make useful examples for a future evaluation set. They are related questions, not a controlled proof that wording alone caused the difference across all settings.

## Corrections to the investigation

### Generalization and citation rules

The system prompt temporarily said to generalize while adding no information absent from the evidence. Generalizing was intended behavior. Evaluation should distinguish a supported broader explanation from an unsupported conclusion, rather than requiring the answer to copy the passage wording.

That edit also left the fragment `using labels such as [1] or [2].` without its explicit citation instruction. Earlier observations correctly noted that citations were absent, but calling them clear instruction-following failures was premature. The appropriate record for those runs is **citations absent; citation instruction unclear**.

The latest user edit restored a request to be specific and cite the evidence. Future citation checks should judge the actual prompt used for each run. Changing these answer rules does not change which passages the retriever selects.

### Manual evidence override

For the diagnostic experiment, a temporary assignment replaced the generated `context` string before the model prompt was built. Automatic retrieval still ran, and the existing source-printing loop still displayed its original results. Those printed results therefore did not identify the evidence sent to the model during the override.

An extra `print(context)` block was suggested to expose the actual supplied evidence. The user later removed the display; inspection of the latest `main.py` also confirms that the manual context assignment is gone and automatic retrieval supplies the answer context again.

The manually selected passages mixed fine-tuning, migration guidance, and multi-vector retrieval. They were relevant but were not a clean, independently validated reference explanation for the everyday question. The experiment cannot establish that a different answer model is necessary.

### Similarity and evaluation

The script normalizes passage and question embeddings, then calculates their dot product. Those values are cosine similarities used to rank passages. Values around `0.64` are not accuracy grades or calibrated relevance probabilities. The implementation always chooses three passages, including when none contains enough evidence.

Manual inspection considered retrieval usefulness, answer accuracy, grounding, and citation support. A repeatable evaluation would need saved questions, verified expected answer points, preserved run settings, and consistent judgments. That work was discussed but is not implemented or reported as completed.

## What the current baseline contains

The current implementation uses 150-word chunks with 30-word overlap; `BAAI/bge-small-en-v1.5` on CPU with batch size 16; top-three cosine retrieval; and `Qwen/Qwen3-0.6B` on CPU in float32 with sampling disabled, thinking disabled in the chat template, and a 256-new-token limit. PyTorch uses eight threads. One hard-coded question runs at a time and passage embeddings are rebuilt on every run.

At documentation time, the local environment was inspected and its package versions captured in `requirements.txt`. Those pins describe the environment now; they do not establish which versions were installed for each historical run. The corpus archive retains source licenses and notices and can be extracted for a local run.

No larger framework, vector database, reranker, persistent index, formal evaluation runner, or cleaned-document layer was added for this milestone. Source display supports manual inspection; it does not validate the generated claims automatically.

## Lessons and stopping point

- **Simplicity:** an understandable script was enough to demonstrate the entire pipeline and expose real failure cases.
- **Fidelity:** a plausible answer and a high-ranked passage are insufficient. Check whether the evidence supports the claim, and preserve commands, versions, warnings, and context when preparing documents.
- **Cost effectiveness:** local models avoided paid inference calls. Time and compute still matter; additional components should serve a specific learning or user need.

The completed milestone is a working local RAG baseline with a supported answer and documented limitations. These principles remain decision criteria; they do not certify reliability. Further evaluation or retrieval improvements would be a new, bounded objective. Designing more complex systems is a possible next direction, with this baseline retained as a concrete reference.
