import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from analysis import NewsAnalyzer, run_analysis
import os

st.set_page_config(page_title="News Analysis Dashboard", layout="wide", page_icon="📰")

@st.cache_data
def load_and_analyze_data(sample_size=5000):
    fake_path = "dataset/Fake.csv"
    true_path = "dataset/True.csv"
    
    if os.path.exists('analyzed_data.csv'):
        df = pd.read_csv('analyzed_data.csv')
        df = df.dropna(subset=['processed_text'])
        df = df[df['processed_text'].str.len() > 0]
        analyzer = NewsAnalyzer()
        topics, _, _ = analyzer.extract_topics_lda(df, n_topics=5)
        keywords = analyzer.extract_tfidf_keywords(df, n_keywords=20)
        stats = analyzer.get_statistics(df)
        return df, topics, keywords, stats, analyzer
    else:
        return run_analysis(fake_path, true_path, sample_size)

st.title("📰 Fake vs True News Analysis Dashboard")
st.markdown("### Sentiment Analysis & Topic Modeling with VADER")

with st.sidebar:
    st.header("Settings")
    sample_size = st.slider("Sample Size", 1000, 10000, 5000, 500)
    
    if st.button("Run Analysis"):
        if os.path.exists('analyzed_data.csv'):
            os.remove('analyzed_data.csv')
        st.cache_data.clear()

with st.spinner("Loading and analyzing data..."):
    df, topics, keywords, stats, analyzer = load_and_analyze_data(sample_size)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💭 Sentiment Analysis", "🔍 Topic Modeling", "☁️ Word Clouds"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", f"{stats['total_articles']:,}")
    col2.metric("Fake News", f"{stats['fake_articles']:,}")
    col3.metric("True News", f"{stats['true_articles']:,}")
    col4.metric("Fake %", f"{(stats['fake_articles']/stats['total_articles']*100):.1f}%")
    
    st.subheader("Distribution of News Types")
    fig = px.pie(
        values=[stats['fake_articles'], stats['true_articles']],
        names=['Fake', 'True'],
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Average Article Length")
    length_data = pd.DataFrame({
        'Type': ['Fake News', 'True News'],
        'Avg Length': [stats['avg_text_length_fake'], stats['avg_text_length_true']]
    })
    fig = px.bar(length_data, x='Type', y='Avg Length', color='Type',
                 color_discrete_map={'Fake News': '#FF6B6B', 'True News': '#4ECDC4'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("VADER Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    col1.metric("Avg Sentiment (Fake)", f"{stats['avg_sentiment_fake']:.3f}")
    col2.metric("Avg Sentiment (True)", f"{stats['avg_sentiment_true']:.3f}")
    
    st.subheader("Sentiment Distribution by News Type")
    fig = px.histogram(df, x='sentiment_compound', color='label',
                      barmode='overlay', nbins=50,
                      color_discrete_map={'fake': '#FF6B6B', 'true': '#4ECDC4'},
                      labels={'sentiment_compound': 'Compound Sentiment Score', 'label': 'News Type'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sentiment Components Comparison")
    sentiment_comp = df.groupby('label')[['sentiment_pos', 'sentiment_neu', 'sentiment_neg']].mean().reset_index()
    sentiment_comp_melted = sentiment_comp.melt(id_vars='label', var_name='Component', value_name='Score')
    
    fig = px.bar(sentiment_comp_melted, x='Component', y='Score', color='label',
                 barmode='group',
                 color_discrete_map={'fake': '#FF6B6B', 'true': '#4ECDC4'},
                 labels={'label': 'News Type'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sentiment vs Text Length")
    sample_df = df.sample(min(1000, len(df)))
    sample_df['text_length'] = sample_df['text'].str.len()
    fig = px.scatter(sample_df, x='text_length', y='sentiment_compound', color='label',
                    color_discrete_map={'fake': '#FF6B6B', 'true': '#4ECDC4'},
                    opacity=0.5,
                    labels={'text_length': 'Text Length', 'sentiment_compound': 'Sentiment'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Topic Modeling (LDA)")
    
    st.subheader("Discovered Topics")
    for topic in topics:
        with st.expander(f"📌 Topic {topic['topic_id'] + 1}"):
            st.write(", ".join(topic['words']))
    
    st.subheader("Top TF-IDF Keywords")
    keywords_df = pd.DataFrame({'Keyword': keywords})
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(keywords_df.iloc[:10], use_container_width=True, hide_index=True)
    with col2:
        st.dataframe(keywords_df.iloc[10:], use_container_width=True, hide_index=True)

with tab4:
    st.header("Word Clouds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fake News")
        fake_text = ' '.join(df[df['label'] == 'fake']['processed_text'].head(500))
        wordcloud_fake = WordCloud(width=800, height=400, background_color='white',
                                   colormap='Reds').generate(fake_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_fake, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.subheader("True News")
        true_text = ' '.join(df[df['label'] == 'true']['processed_text'].head(500))
        wordcloud_true = WordCloud(width=800, height=400, background_color='white',
                                   colormap='Blues').generate(true_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_true, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard analyzes fake vs true news using:\n"
    "- **VADER** for sentiment analysis\n"
    "- **LDA** for topic modeling\n"
    "- **TF-IDF** for keyword extraction"
)

with st.expander("📄 Sample Articles"):
    article_type = st.radio("Select type:", ["fake", "true"])
    sample_articles = df[df['label'] == article_type].head(3)
    
    for idx, row in sample_articles.iterrows():
        st.markdown(f"**{row['title']}**")
        st.write(row['text'][:300] + "...")
        st.write(f"Sentiment: {row['sentiment_compound']:.3f}")
        st.markdown("---")
