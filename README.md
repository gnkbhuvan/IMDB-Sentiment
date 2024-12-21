# IMDb Sentiment Analysis Classifier

This project implements a sentiment analysis classifier using the IMDb movie reviews dataset. The classifier can determine whether a movie review expresses positive or negative sentiment.

## Features

- Text preprocessing including:
  - Tokenization
  - Lowercasing
  - Stop words removal
  - Special characters removal
  - Stemming
- TF-IDF vectorization for text representation
- Logistic Regression classifier
- Performance evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix visualization

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have the IMDb dataset in the `aclImdb` directory with the following structure:
```
aclImdb/
├── train/
│   ├── pos/
│   └── neg/
├── test/
│   ├── pos/
│   └── neg/
└── README
```

## Usage

Run the sentiment analyzer:
```bash
python sentiment_analyzer.py
```

This will:
1. Load and preprocess the IMDb dataset
2. Train a sentiment classification model
3. Evaluate the model's performance
4. Save a confusion matrix visualization
5. Run a sample prediction

## Model Details

- **Vectorization**: TF-IDF with 5000 features
- **Classifier**: Logistic Regression
- **Train/Validation Split**: 80/20

## Performance

The model's performance metrics will be displayed after training, including:
- Accuracy score
- Detailed classification report
- Confusion matrix visualization (saved as 'confusion_matrix.png')
