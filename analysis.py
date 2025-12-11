import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class NewsAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        additional_stops = {'said', 'would', 'one', 'us', 'also', 'like', 'get', 'new', 'may', 'two', 'could', 'mr', 'ms', 'even', 'much', 'percent', 'according'}
        self.stop_words.update(additional_stops)
        self.vader = SentimentIntensityAnalyzer()
        
    def load_data(self, fake_path, true_path, sample_size=None):
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 'fake'
        
        true_df = pd.read_csv(true_path)
        true_df['label'] = 'true'
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        return df
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def apply_preprocessing(self, df):
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df['processed_title'] = df['title'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 0]
        return df
    
    def analyze_sentiment(self, df):
        sentiments = []
        for text in df['text']:
            if pd.isna(text):
                sentiments.append({'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0})
            else:
                sentiments.append(self.vader.polarity_scores(str(text)))
        
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        df['sentiment_pos'] = [s['pos'] for s in sentiments]
        df['sentiment_neu'] = [s['neu'] for s in sentiments]
        df['sentiment_neg'] = [s['neg'] for s in sentiments]
        
        return df
    
    def extract_topics_lda(self, df, n_topics=5, n_words=10):
        df_clean = df[df['processed_text'].notna() & (df['processed_text'].str.len() > 0)].copy()
        vectorizer = CountVectorizer(max_features=2000, max_df=0.7, min_df=10, stop_words=list(self.stop_words))
        doc_term_matrix = vectorizer.fit_transform(df_clean['processed_text'])
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20, learning_method='batch')
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words
            })
        
        return topics, lda, vectorizer
    
    def extract_tfidf_keywords(self, df, n_keywords=20):
        df_clean = df[df['processed_text'].notna() & (df['processed_text'].str.len() > 0)].copy()
        tfidf = TfidfVectorizer(max_features=n_keywords, ngram_range=(1, 2))
        tfidf.fit(df_clean['processed_text'])
        
        feature_names = tfidf.get_feature_names_out()
        return feature_names.tolist()
    
    def get_statistics(self, df):
        stats = {
            'total_articles': len(df),
            'fake_articles': len(df[df['label'] == 'fake']),
            'true_articles': len(df[df['label'] == 'true']),
            'avg_sentiment_fake': df[df['label'] == 'fake']['sentiment_compound'].mean(),
            'avg_sentiment_true': df[df['label'] == 'true']['sentiment_compound'].mean(),
            'avg_text_length_fake': df[df['label'] == 'fake']['text'].str.len().mean(),
            'avg_text_length_true': df[df['label'] == 'true']['text'].str.len().mean(),
        }
        return stats

def run_analysis(fake_path, true_path, sample_size=5000):
    analyzer = NewsAnalyzer()
    
    print("Loading data...")
    df = analyzer.load_data(fake_path, true_path, sample_size)
    
    print("Preprocessing text...")
    df = analyzer.apply_preprocessing(df)
    
    print("Analyzing sentiment...")
    df = analyzer.analyze_sentiment(df)
    
    print("Extracting topics...")
    topics, lda_model, vectorizer = analyzer.extract_topics_lda(df, n_topics=5)
    
    print("Extracting keywords...")
    keywords = analyzer.extract_tfidf_keywords(df, n_keywords=20)
    
    print("Calculating statistics...")
    stats = analyzer.get_statistics(df)
    
    return df, topics, keywords, stats, analyzer

if __name__ == "__main__":
    fake_path = "dataset/Fake.csv"
    true_path = "dataset/True.csv"
    
    df, topics, keywords, stats, analyzer = run_analysis(fake_path, true_path)
    
    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Top Topics ===")
    for topic in topics:
        print(f"Topic {topic['topic_id']}: {', '.join(topic['words'][:5])}")
    
    df.to_csv('analyzed_data.csv', index=False)
    print("\nAnalysis complete! Results saved to 'analyzed_data.csv'")
