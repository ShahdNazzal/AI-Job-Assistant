# Retrieval-Augmented Generation (RAG)

## The problem RAG solves
Large language models generate fluent text but only know what was in their training data or prompt. Asking them about your private documents invites hallucination. Retrieval-Augmented Generation fixes this: first retrieve relevant passages from a trusted corpus, then pass them to the LLM as context and instruct it to answer only from that evidence, citing sources.

## The standard pipeline
1. Ingest: collect knowledge documents (Markdown, PDFs, wiki pages).
2. Chunk: split them into paragraph-sized passages; heading-based chunking keeps context. Chunks that are too small lose meaning; chunks that are too large dilute retrieval precision.
3. Embed: convert each chunk into a vector with a sentence-embedding model.
4. Index: store vectors in memory (NumPy, FAISS) or a vector database.
5. Retrieve: embed the user's question and rank chunks by cosine similarity; keep the top-k above a minimum score.
6. Generate: build a prompt containing the question, the retrieved excerpts, and strict grounding rules; ask the model to cite sources like [KB: title].
7. Evaluate: check retrieval precision on labeled questions and whether answers actually follow from the cited chunks.

## Grounding rules that matter
A grounded system must refuse to answer when retrieval returns nothing relevant, must distinguish document evidence from model speculation, and must never present a draft as a verified fact. Displaying the retrieved sources next to the answer - as this application does - lets users verify claims themselves.

## Common failure modes
Retrieval misses happen when questions use different vocabulary than the documents; adding synonyms or hybrid keyword search helps. Generation failures happen when the prompt is too permissive; short, explicit rules outperform long polite requests. Finally, chunking bugs (broken sentences, lost headings) silently degrade everything downstream, so inspect chunks by hand before blaming the model.

## When RAG is overkill
If the knowledge fits in one page, just paste it into the prompt. RAG earns its complexity when the corpus is larger than the context window, changes independently of the code, or must be filtered per user. A small portfolio app can implement the full loop with NumPy alone - no vector database required.
