# Sentiment Analysis with IMDB Dataset

A machine learning project that performs sentiment analysis on IMDB movie reviews using a Naive Bayes classifier.

## Project Overview

This project builds a sentiment classification model to predict whether a movie review is positive or negative. It uses the IMDB Dataset and applies natural language processing techniques combined with machine learning to achieve high accuracy.

## Features

- **Dataset**: IMDB Dataset with movie reviews and sentiment labels
- **Text Preprocessing**: Removes missing values and standardizes text data
- **Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features
- **Classification Model**: Trained Multinomial Naive Bayes classifier
- **Evaluation**: Accuracy metrics and visualization

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone or download the project
2. Install required dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Dataset

The project uses `IMDB_Dataset.csv` which contains:
- `review`: Text content of movie reviews
- `sentiment`: Classification label (positive/negative)

## Usage

Run the sentiment analysis script:

```bash
python sentiment_analysis.py
```

The script will:
1. Load and clean the IMDB dataset
2. Vectorize text data using TF-IDF
3. Split data into training (80%) and testing (20%) sets
4. Train a Multinomial Naive Bayes model
5. Generate predictions on test data
6. Display model accuracy and generate a visualization

## Model Performance

The script outputs:
- **Accuracy Score**: Overall accuracy percentage on test set
- **Accuracy Visualization**: Bar chart showing model performance

## Project Structure

```
sentiment_analysis/
├── sentiment_analysis.py      # Main script
├── IMDB_Dataset.csv          # Dataset file
└── README.md                  # This file
```

## How It Works

1. **Data Loading**: Reads IMDB dataset from CSV file
2. **Data Cleaning**: Removes null values and standardizes review text
3. **Feature Extraction**: TF-IDF vectorizer converts text reviews to numerical vectors
4. **Model Training**: Trains Naive Bayes classifier on 80% of data
5. **Evaluation**: Tests model on remaining 20% and calculates accuracy

## Notes

- The random state is set to 42 for reproducible results
- Stop words are removed during text vectorization
- Train-test split uses 80-20 ratio with stratification
