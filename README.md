# IMDB Sentiment Analysis

A machine learning project that performs sentiment analysis on IMDB movie reviews using both traditional ML and deep learning approaches.

## Live Demo
Try out the application here: [IMDB Sentiment Analysis App](https://imdb-data-sentiment-analysis.streamlit.app)

## Features
- Text preprocessing with lemmatization and POS tagging
- Multiple model implementations:
  - Traditional ML: Logistic Regression and SVM with TF-IDF
  - Deep Learning: Fine-tuned BERT model
- Interactive web interface using Streamlit
- Real-time sentiment prediction

## Project Structure
```
├── sentiment_analyzer.py   # Main ML implementation
├── app.py                 # Streamlit web application
├── imdb_trainer.ipynb     # BERT model implementation
├── requirements.txt       # Project dependencies
└── download_nltk_data.py  # NLTK data downloader
```

## Setup and Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data:
   ```bash
   python download_nltk_data.py
   ```

## Usage
1. Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```
2. Enter a movie review text
3. Get instant sentiment prediction with confidence score

## Models
- **Traditional ML Pipeline**:
  - TF-IDF vectorization
  - Logistic Regression and SVM classifiers
  - GridSearchCV for hyperparameter optimization

- **BERT Implementation**:
  - Fine-tuned BERT-base-uncased
  - Handles nuanced sentiment expressions
  - Better context understanding

## Performance
- Fast inference time
- High accuracy on IMDB dataset
- Robust handling of various review styles

## Online Deployment
The application is deployed using Streamlit Cloud and is accessible at:
[https://imdb-data-sentiment-analysis.streamlit.app](https://imdb-data-sentiment-analysis.streamlit.app)

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
This project is open source and available under the MIT License.
