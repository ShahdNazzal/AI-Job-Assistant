# NLP and Embeddings

## From words to vectors
Text must become numbers before models can learn from it. Early approaches used bag-of-words and TF-IDF: simple, strong baselines for classification. Word embeddings such as word2vec and GloVe encode words as dense vectors where similar words sit close together. Modern sentence embeddings, produced by transformer models like all-MiniLM-L6-v2, encode whole sentences or paragraphs so that semantically similar texts have high cosine similarity even when wording differs.

## What cosine similarity measures
Cosine similarity compares the angle between two vectors, ignoring magnitude. With normalized embeddings it ranges from -1 to 1, and in practice meaningful texts land between roughly 0.2 and 0.9. High cosine similarity means the texts discuss related content; it does not mean facts are true, skills are equivalent, or a candidate is qualified. Always treat semantic scores as evidence to inspect, not verdicts.

## Transformer models in practice
Transformers process tokens with self-attention, letting every token attend to every other one. Pretrained models from Hugging Face can be fine-tuned on small labeled datasets for classification, named entity recognition, or question answering. Fine-tuning a small model on a few thousand well-labeled examples often beats prompting a giant model for narrow, stable tasks - and it is cheaper to run.

## Practical NLP project checklist
Define the task and metric before writing code. Inspect at least 100 raw examples by hand. Build the simplest baseline first (TF-IDF + logistic regression), then a transformer, and compare fairly on the same split. Keep a confusion matrix and read the errors: most insight lives there. Report the limitations honestly - language coverage, domain shift, and label noise.

## Where embeddings fail
Embeddings can miss negation ("supports" vs "does not support"), numeric details ("2 years" vs "5 years"), and rare domain jargon. They also encode biases present in training data. When precision matters, combine semantic retrieval with explicit rules or dictionary matching - exactly the hybrid used by this project.
