# Custom Sentiment Analysis Streamlit App
# Save as: streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Page config
st.set_page_config(
    page_title="Custom Sentiment AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        color: #856404;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CustomSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.load_model_components()
    
    @st.cache_resource
    def load_model_components(_self):
        """Load custom trained model"""
        try:
            _self.model = load_model('best_sentiment_model.h5')
            
            with open('tokenizer.pkl', 'rb') as f:
                _self.tokenizer = pickle.load(f)
            
            with open('model_config.pkl', 'rb') as f:
                _self.config = pickle.load(f)
            
            st.success("✅ Custom trained model loaded successfully!")
            return True
        except:
            st.warning("⚠️ Model files not found. Run Jupyter notebook first to train model.")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            return text
    
    def predict_sentiment(self, text):
        """Predict sentiment using custom model"""
        if not text or not text.strip():
            return "neutral", 0.5, {"neutral": 0.5, "positive": 0.25, "negative": 0.25}
        
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return "neutral", 0.5, {"neutral": 0.5, "positive": 0.25, "negative": 0.25}
        
        if self.model and self.tokenizer and self.config:
            try:
                sequence = self.tokenizer.texts_to_sequences([cleaned_text])
                padded_sequence = pad_sequences(sequence, 
                                               maxlen=self.config['max_sequence_length'], 
                                               padding='post')
                
                prediction = self.model.predict(padded_sequence, verbose=0)
                
                if self.config['is_binary']:
                    confidence = float(prediction[0][0])
                    sentiment = "positive" if confidence > 0.5 else "negative"
                    confidence = confidence if sentiment == "positive" else 1 - confidence
                    scores = {"positive": confidence, "negative": 1-confidence}
                else:
                    class_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_idx])
                    sentiment = self.config['target_names'][class_idx].lower()
                    scores = {name.lower(): float(score) for name, score in 
                             zip(self.config['target_names'], prediction[0])}
                
                return sentiment, confidence, scores
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        
        return "neutral", 0.5, {"neutral": 0.5, "positive": 0.25, "negative": 0.25}

# Initialize analyzer
@st.cache_resource
def load_analyzer():
    return CustomSentimentAnalyzer()

analyzer = load_analyzer()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Header
st.markdown('<h1 class="main-header">🧠 Custom Sentiment Analysis AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2em;">Powered by Custom Neural Networks trained from scratch on Kaggle datasets</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Model Info")
    
    if analyzer.config:
        st.success(f"**Model**: {analyzer.config['model_name']}")
        st.info(f"**Accuracy**: {analyzer.config['test_accuracy']:.1%}")
        st.info(f"**Classes**: {analyzer.config['num_classes']}")
    else:
        st.error("Model not loaded. Train model first!")
    
    st.markdown("---")
    analysis_mode = st.selectbox(
        "Analysis Mode:",
        ["Single Text", "Batch Analysis", "Live Stream"]
    )

# Main content
if analysis_mode == "Single Text":
    st.header("📝 Single Text Analysis")
    
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here...",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    with col2:
        real_time = st.toggle("Real-time analysis")
    
    if user_input and (analyze_button or real_time):
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence, scores = analyzer.predict_sentiment(user_input)
        
        # Display result
        if sentiment == 'positive':
            st.markdown(f"""
            <div class="sentiment-positive">
                <h2>😊 Positive Sentiment</h2>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        elif sentiment == 'negative':
            st.markdown(f"""
            <div class="sentiment-negative">
                <h2>😢 Negative Sentiment</h2>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sentiment-neutral">
                <h2>😐 Neutral Sentiment</h2>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence scores chart
        if scores:
            fig = go.Figure(data=go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=['#ff6b6b' if k == 'negative' else '#4ecdc4' if k == 'positive' else '#feca57' for k in scores.keys()]
            ))
            fig.update_layout(
                title="Confidence Scores",
                xaxis_title="Sentiment",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Add to history
        st.session_state.analysis_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text': user_input[:100] + '...' if len(user_input) > 100 else user_input,
            'sentiment': sentiment,
            'confidence': confidence
        })

elif analysis_mode == "Batch Analysis":
    st.header("📊 Batch Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:", df.head())
        
        text_column = st.selectbox("Select text column:", df.columns.tolist())
        
        if st.button("Analyze All"):
            progress_bar = st.progress(0)
            results = []
            
            for idx, text in enumerate(df[text_column]):
                if pd.notna(text):
                    sentiment, confidence, scores = analyzer.predict_sentiment(str(text))
                    results.append({
                        'text': str(text)[:100] + '...',
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", len(results_df[results_df['sentiment'] == 'positive']))
            with col2:
                st.metric("Negative", len(results_df[results_df['sentiment'] == 'negative']))
            with col3:
                st.metric("Neutral", len(results_df[results_df['sentiment'] == 'neutral']))
            
            # Results table
            st.dataframe(results_df)
            
            # Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )

elif analysis_mode == "Live Stream":
    st.header("🔴 Live Sentiment Stream")
    
    stream_text = st.text_input("Type for real-time analysis:")
    
    if stream_text:
        sentiment, confidence, scores = analyzer.predict_sentiment(stream_text)
        
        col1, col2 = st.columns(2)
        with col1:
            if sentiment == 'positive':
                st.success(f"😊 POSITIVE ({confidence:.1%})")
            elif sentiment == 'negative':
                st.error(f"😢 NEGATIVE ({confidence:.1%})")
            else:
                st.warning(f"😐 NEUTRAL ({confidence:.1%})")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")

# History display
if st.session_state.analysis_history:
    st.subheader("📈 Analysis History")
    history_df = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(history_df)
    
    if st.button("Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🧠 Custom Sentiment Analysis AI</strong></p>
    <p>Built with TensorFlow, Keras, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
