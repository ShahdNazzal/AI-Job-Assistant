# Vector Search and Vector Databases

## What vector search does
Vector search finds the items whose embeddings are most similar to a query embedding. The index is a matrix of vectors; the query is embedded with the same model; similarity is usually cosine (equivalently the dot product of normalized vectors). Exact search with NumPy is simply scores = vectors @ query - perfectly fine up to tens of thousands of chunks.

## Approximate nearest neighbors at scale
When the corpus grows to millions of vectors, exact search becomes slow, and approximate indexes trade a little recall for a lot of speed. FAISS provides IVF (cluster-then-probe) and HNSW (graph-based) indexes. Tuning means measuring recall@k against exact search on a sample, not just latency.

## Choosing a storage option
For a portfolio project, an in-memory NumPy matrix rebuilt at startup is transparent, fast, and dependency-free. For persistence, SQLite plus a serialized matrix works. Managed vector databases (Pinecone, Weaviate, Qdrant, Milvus, Chroma) add filtering, hybrid keyword search, replication, and multi-tenant support - valuable in production, unnecessary for demos. PostgreSQL with the pgvector extension is a popular middle ground: vectors live next to relational data, with SQL filtering.

## Filtering and hybrid search
Real questions often need constraints: filter by document type, date, or tenant before ranking. Most vector stores support metadata filters. Hybrid search combines BM25 keyword scores with vector similarity, which helps with rare terms and exact names that embeddings blur.

## Practical hygiene
Always embed queries with the same model and preprocessing as the documents. Keep the embedding model version next to the vectors: mixing versions silently destroys similarity. Monitor retrieval quality with a small labeled set of question-to-source pairs; if retrieval is wrong, no amount of LLM prompt engineering will save the answer.
