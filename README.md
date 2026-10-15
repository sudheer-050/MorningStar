# MorningStar

![MorningStar NLP classification pipeline](docs/images/morningstar-nlp.png)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-scikit--learn-3776AB?logo=python&logoColor=white)

A DePaul University capstone project: an NLP pipeline that classifies company business-segment descriptions into Morningstar's GECS (Global Equity Classification Structure) — 145 hierarchical industry categories — using TF-IDF features and a LinearSVC classifier.

## What it does

- **Knowledge-based imputation** — fills in missing segment descriptions using a lookup table of official industry definitions, instead of dropping or blindly imputing rows.
- **Stratification guard** — automatically drops industry classes with fewer than 2 samples so the train/test split can safely stratify on the target.
- **Text-only TF-IDF + LinearSVC pipeline** — 1–3 gram TF-IDF features (with domain-specific stop words like "company", "segment") feeding a class-weight-balanced linear SVM.
- **Descriptive analytics** — prints top/bottom 10 industries by F1-score and overall accuracy/macro-F1 after training.
- **Interactive recommendation engine** — after training, type a free-text business description and get the top-3 most likely GECS industries with calibrated confidence percentages (temperature-scaled softmax over the SVM's decision scores).

## Files

| File | What it covers |
|---|---|
| `task1-exp5 - final got 0.7901.py` | Task 1 baseline: text + engineered numeric features, macro F1 ≈ 0.79 |
| `task1-exp10-new with filter- f1-0.8275.py` | Task 1 improved: text-only pipeline with a revenue-share filter and stronger imputation, macro F1 ≈ 0.83 |
| `task2-exp5- with 1 &2 accurate answers.py` | Task 2: extends Task 1 into hierarchical sub-industry (activity-level) classification, with combined Task 1 + Task 2 performance reporting |

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## Data

This project was built against a proprietary Morningstar GECS dataset provided as part of a DePaul University capstone course and isn't included in this repo. The scripts reference local file paths from that coursework environment — they're shared here for the modeling approach and methodology, not as a plug-and-run tool. To use them, point the `pd.read_csv(...)` calls near the top of each script at your own equivalent dataset (a GECS-labeled table of business segment descriptions, plus an industry-code lookup table).

## License

MIT — see [LICENSE](LICENSE).
