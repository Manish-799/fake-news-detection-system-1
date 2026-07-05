# Dataset

This project uses the **Fake and Real News Dataset** containing labelled
real and fake news articles.

## Expected Files
Place the following files in this directory before training:

```text
dataset/
├── True.csv
├── Fake.csv
└── README.md
```

The training pipeline expects both CSV files to contain at least:

- `title`
- `text`

## Labels

The training script assigns:

0 = Fake  
1 = Real

## Preprocessing

The training pipeline:

- Combines the article title and body.
- Applies shared text preprocessing.
- Removes blank articles.
- Removes exact duplicates based on normalized article text.
- Performs a stratified train-test split.
- Learns TF-IDF features from the training split.

## Dataset Files

The CSV dataset files are not committed to this repository.

Download the **Fake and Real News Dataset** from Kaggle and place `True.csv` and `Fake.csv` in this directory before retraining the model.

## Limitations

The dataset is dominated by a specific historical and political news distribution.

The two labelled classes also contain differences in publication style and editorial language. As a result, performance on unseen publishers, geographic domains, or newer news cycles may differ from held-out test performance.

The classifier learns statistical linguistic patterns. It does not independently verify factual claims.
