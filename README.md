# Fake vs True News Analysis

NLP analysis pipeline with VADER sentiment analysis and interactive dashboard.

## Features

- Text preprocessing (lowercasing, stopword removal, tokenization)
- Sentiment analysis using VADER
- Topic modeling with LDA
- TF-IDF keyword extraction
- Interactive Streamlit dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Analysis Script
```bash
python analysis.py
```

### Launch Dashboard
```bash
streamlit run app.py
```

## Project Structure

- `dataset/` - Contains Fake.csv and True.csv
- `analysis.py` - Core analysis and preprocessing
- `app.py` - Streamlit dashboard
- `requirements.txt` - Dependencies
