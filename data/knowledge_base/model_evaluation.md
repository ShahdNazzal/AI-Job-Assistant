# Model and Application Evaluation

## Why evaluation is an engineering skill
Saying "the model works" is not engineering. Quantify behavior with metrics computed from defined examples, keep the evaluation code in the repository, and rerun it on every change. A tiny honest evaluation - ten labeled cases with precision, recall, and F1 - is worth more than an unspecified "high accuracy" claim.

## Classification metrics
For a binary classifier: precision is TP / (TP + FP) - of everything predicted positive, the share that is right. Recall is TP / (TP + FN) - of everything truly positive, the share you found. F1 is their harmonic mean and punishes imbalance between them. Compute them micro-averaged (pool all decisions) when overall behavior matters, macro-averaged when each class matters equally.

## Evaluating extraction components
Skill extraction is a set-prediction task: compare the predicted skill set with a labeled set. True positives are predicted-and-expected, false positives are predicted-but-wrong, false negatives are expected-but-missed. Micro-averaged precision and recall follow directly. Watch boundary errors: matching "Java" inside "JavaScript" is a false positive that a good evaluation will catch.

## Evaluating matching and ranking
For similarity-based systems, test relative expectations rather than absolute numbers: a strong-matching resume/job pair should score above a weak pair, and scores must stay in valid ranges. Encode these as labeled cases with expected tiers and check ordering. This survives model updates better than pinning exact score values.

## Honest reporting
State the dataset size and its origin, avoid generalizing from toy data, and never report a metric you did not compute in this session. When a component cannot be evaluated - say the embedding model is unavailable - display that fact instead of a number. In production you would add inter-annotator agreement, larger labeled corpora, and regression tests wired into CI.
