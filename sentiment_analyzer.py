import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# NLTK imports for lemmatization
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

class SentimentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.lemmatizer = WordNetLemmatizer()
        
        # Create two pipelines: one for Logistic Regression and one for SVM
        self.lr_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        self.svm_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', SVC(random_state=42, probability=True))
        ])
        
        # Define parameter grids for both models
        self.lr_param_grid = {
            'vectorizer__max_features': [5000],
            'vectorizer__ngram_range': [(1, 2)],
            'classifier__C': [1.0],
            'classifier__class_weight': ['balanced']
        }
        
        self.svm_param_grid = {
            'vectorizer__max_features': [5000],
            'vectorizer__ngram_range': [(1, 2)],
            'classifier__C': [1.0],
            'classifier__kernel': ['linear'],
            'classifier__class_weight': ['balanced']
        }
        
        # The best pipeline will be stored here after training
        self.best_pipeline = None
        self.best_model_name = None
    
    def get_wordnet_pos(self, word):
        """Map POS tag to first character used by WordNetLemmatizer"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatization with POS tagging
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) 
            for word in tokens
        ]
        
        return ' '.join(lemmatized_tokens)
    
    def load_data(self):
        reviews = []
        labels = []
        
        # Load positive reviews
        pos_path = os.path.join(self.data_path, 'train', 'pos')
        for filename in tqdm(os.listdir(pos_path), desc='Loading positive reviews'):
            with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1)
        
        # Load negative reviews
        neg_path = os.path.join(self.data_path, 'train', 'neg')
        for filename in tqdm(os.listdir(neg_path), desc='Loading negative reviews'):
            with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(0)
        
        return reviews, labels
    
    def prepare_data(self):
        print("Loading data...")
        reviews, labels = self.load_data()
        
        print("Preprocessing texts...")
        processed_reviews = [self.preprocess_text(review) for review in tqdm(reviews)]
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            processed_reviews, labels, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self):
        print("Preparing data...")
        X_train, X_val, y_train, y_val = self.prepare_data()
        
        # Train and evaluate Logistic Regression
        print("\nTraining Logistic Regression model...")
        lr_grid_search = GridSearchCV(
            self.lr_pipeline,
            self.lr_param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        lr_grid_search.fit(X_train, y_train)
        lr_score = lr_grid_search.score(X_val, y_val)
        
        # Train and evaluate SVM
        print("\nTraining SVM model...")
        svm_grid_search = GridSearchCV(
            self.svm_pipeline,
            self.svm_param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        svm_grid_search.fit(X_train, y_train)
        svm_score = svm_grid_search.score(X_val, y_val)
        
        # Compare models and select the best one
        print("\nModel Comparison:")
        print(f"Logistic Regression Validation Accuracy: {lr_score:.4f}")
        print(f"SVM Validation Accuracy: {svm_score:.4f}")
        
        # Select the best model
        if lr_score >= svm_score:
            self.best_pipeline = lr_grid_search.best_estimator_
            self.best_model_name = "Logistic Regression"
            best_params = lr_grid_search.best_params_
            best_score = lr_score
        else:
            self.best_pipeline = svm_grid_search.best_estimator_
            self.best_model_name = "SVM"
            best_params = svm_grid_search.best_params_
            best_score = svm_score
        
        print(f"\nBest Model: {self.best_model_name}")
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        print(f"\nBest validation accuracy: {best_score:.4f}")
        
        # Generate detailed report for the best model
        y_pred = self.best_pipeline.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        class_report = classification_report(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Save metrics to file
        with open('model_metrics.txt', 'w') as f:
            f.write(f"Best Model: {self.best_model_name}\n\n")
            f.write("Best parameters found:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nBest validation accuracy: {best_score:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(class_report)
        
        return accuracy, class_report, conf_matrix
    
    def save_model(self, model_path=None):
        """Save the trained model to disk"""
        if model_path is None:
            # Generate a filename with timestamp and model type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'sentiment_model_{self.best_model_name}_{timestamp}.joblib'
        
        if self.best_pipeline is None:
            raise ValueError("No trained model found. Please train the model first.")
        
        joblib.dump(self.best_pipeline, model_path)
        print(f"Model saved to {model_path}")
        
        # Save a reference to the latest model
        joblib.dump(model_path, 'latest_model.txt')
        return model_path
    
    @classmethod
    def load_model(cls, model_path=None):
        """Load a trained model from disk"""
        if model_path is None:
            # Load the path to the latest model
            try:
                model_path = joblib.load('latest_model.txt')
            except:
                raise ValueError("No model path provided and no latest model found.")
        
        analyzer = cls(None)  # Create instance without data path
        analyzer.best_pipeline = joblib.load(model_path)
        return analyzer
    
    def predict(self, text):
        if self.best_pipeline is None:
            raise ValueError("No trained model found. Please train or load a model first.")
        
        processed_text = self.preprocess_text(text)
        prediction = self.best_pipeline.predict([processed_text])[0]
        probability = self.best_pipeline.predict_proba([processed_text])[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return sentiment, confidence

if __name__ == "__main__":
    # Initialize and train the model
    analyzer = SentimentAnalyzer('aclImdb')
    analyzer.train_model()
    
    # Save the trained model
    model_path = analyzer.save_model()
    
    # Example predictions
    test_texts = [
        "This movie was fantastic! I really enjoyed every moment of it.",
        "The plot was confusing and the acting was terrible.",
        "A masterpiece of modern cinema with brilliant performances.",
        "I fell asleep during the movie, it was so boring."
    ]
    
    print("\nSample Predictions:")
    for text in test_texts:
        sentiment, confidence = analyzer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
