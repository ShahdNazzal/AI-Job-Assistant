# Interview Preparation for AI Engineer Roles

## What interviewers actually test
For junior AI roles, interviewers probe three layers: ML fundamentals (bias-variance, overfitting, precision/recall, cross-validation), coding (clean Python, data manipulation, complexity), and system sense (how you would serve and monitor a model). Deep research knowledge is rarely expected; clear reasoning and honesty about unknowns are.

## Explaining your projects
Prepare a two-minute structure for each project: the problem, your specific contribution, the evaluation, and one limitation you would fix next. Numbers you measured yourself - dataset size, F1, latency - carry weight; do not present numbers you cannot reproduce. Expect follow-ups like "why this metric?" and "what breaks at 100x data?".

## Common technical questions
- Why is accuracy misleading on imbalanced data, and what do you use instead?
- Explain precision versus recall with an example you have built.
- What is data leakage and how do Pipelines prevent it?
- How does cosine similarity behave for embeddings, and when does it mislead?
- How would you reduce hallucinations in an LLM feature? (Grounding, citations, refusal paths, evaluation.)
- How would you deploy a model and monitor it for drift?

## Practical rounds
You may be asked to parse a text file, build a small API, or compute similarity between documents. Practice with the same tools this project uses: PyMuPDF/python-docx for extraction, sentence-transformers for embeddings, FastAPI for serving, pytest for tests. Narrate your decisions; silent coding loses easy points.

## Questions to ask them
Ask how models reach production, who owns monitoring, how experiments are reviewed, and what the on-call story is for AI features. Thoughtful questions signal that you have operated systems, not just trained models.
