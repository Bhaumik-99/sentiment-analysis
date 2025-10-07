 import pandas as pd 
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class Sentiment140Trainer:
    def __init__(self):
        self.models = {}:
        self.vectorizer = None   
        self.best_model = None
        self.best_model_name = None
        
        # Common English stopwords
        self.stop_words = { 
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
            'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
            'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
            'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
            'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
            'wouldn', "wouldn't"
        }
        
    def load_data(self, filepath, sample_size=None):
        """Load the Sentiment140 dataset"""
        print("Loading dataset...")
        
        # Column names for the Sentiment140 dataset
        columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        
        try:
            # Load dataset
            df = pd.read_csv(filepath, encoding='latin-1', names=columns)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Sample data if specified (for faster training during development)
            if sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"Using sample of {sample_size} records")
            
            # Convert target labels (0=negative, 4=positive) to binary (0=negative, 1=positive)
            df['sentiment'] = df['target'].map({0: 0, 4: 1})
            
            # Remove rows with missing sentiment mapping
            df = df.dropna(subset=['sentiment'])
            
            print(f"Final dataset shape: {df.shape}")
            print("Sentiment distribution:")
            print(df['sentiment'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure the CSV file is in the correct path and has the expected format")
            return None
    
    def simple_tokenize(self, text):
        """Simple tokenization without external dependencies"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def preprocess_text(self, text):
        """Clean and preprocess text without NLTK dependencies"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove RT (retweet) markers
        text = re.sub(r'\brt\b', '', text)
        
        # Handle contractions and common abbreviations
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "there's": "there is"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        
        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple tokenization
        tokens = self.simple_tokenize(text)
        
        # Remove stopwords and short tokens
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(filtered_tokens)
    
    def prepare_data(self, df, test_size=0.2, validation_size=0.1):
        """Prepare data for training"""
        print("Preprocessing text data...")
        
        # Apply text preprocessing
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"After preprocessing, dataset shape: {df.shape}")
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['sentiment']
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, X_val, y_train, y_val, max_features=10000):
        """Train multiple models and compare performance"""
        print("Training multiple models...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Fit vectorizer on training data
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Linear SVM': LinearSVC(
                random_state=42, 
                max_iter=2000,
                C=1.0,
                dual=False
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=20,
                min_samples_split=5
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train_vec, y_train)
                
                # Predictions
                train_pred = model.predict(X_train_vec)
                val_pred = model.predict(X_val_vec)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                # Store model and results
                self.models[name] = model
                results[name] = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'val_predictions': val_pred
                }
                
                print(f"{name} - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not results:
            print("No models were successfully trained!")
            return None
            
        # Find best model
        best_val_acc = max(results.values(), key=lambda x: x['val_accuracy'])['val_accuracy']
        self.best_model_name = [name for name, res in results.items() 
                               if res['val_accuracy'] == best_val_acc][0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name} with validation accuracy: {best_val_acc:.4f}")
        
        return results
    
    def evaluate_model(self, X_test, y_test, model_name=None):
        """Evaluate model on test set"""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        print(f"\nEvaluating {model_name} on test set...")
        
        # Transform test data
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # ROC AUC if available
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                print(f"ROC AUC Score: {auc_score:.4f}")
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test_vec)
                auc_score = roc_auc_score(y_test, y_scores)
                print(f"ROC AUC Score: {auc_score:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
        
        return accuracy, y_pred
    
    def plot_model_comparison(self, results):
        """Plot comparison of different models"""
        if not results:
            print("No results to plot")
            return
            
        models = list(results.keys())
        train_accs = [results[model]['train_accuracy'] for model in models]
        val_accs = [results[model]['val_accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8)
        plt.bar(x + width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_sentiment(self, texts):
        """Predict sentiment for new texts"""
        if self.best_model is None or self.vectorizer is None:
            print("No trained model available. Please train a model first.")
            return None
        
        # Preprocess texts
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X_vec = self.vectorizer.transform(processed_texts)
        
        # Predict
        predictions = self.best_model.predict(X_vec)
        
        # Get probabilities if available
        try:
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X_vec)
            elif hasattr(self.best_model, 'decision_function'):
                scores = self.best_model.decision_function(X_vec)
                # Convert to probabilities using sigmoid
                probabilities = 1 / (1 + np.exp(-scores))
                probabilities = np.column_stack([1 - probabilities, probabilities])
            else:
                probabilities = None
        except:
            probabilities = None
        
        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            result = {
                'text': text,
                'sentiment': 'Positive' if pred == 1 else 'Negative',
                'confidence': probabilities[i][pred] if probabilities is not None else None
            }
            results.append(result)
        
        return results
    
    def save_model(self, filepath):
        """Save the trained model and vectorizer"""
        model_data = {
            'vectorizer': self.vectorizer,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'all_models': self.models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a previously trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.models = model_data['all_models']
            
            print(f"Model loaded successfully. Best model: {self.best_model_name}")
            
        except Exception as e:
            print(f"Error loading model: {e}")

# Main execution function
def main():
    """Main function to demonstrate the training process"""
    
    # Initialize trainer
    trainer = Sentiment140Trainer()
    
    # Configuration
    DATA_PATH = r"C:\Users\ravi5\sentiment-analysis\training.1600000.processed.noemoticon.csv"
    SAMPLE_SIZE = 1000000  # Start with 10k for testing, use None for full dataset
    MODEL_SAVE_PATH = "sentiment_model.pkl"
    
    print("=== Sentiment Analysis Model Training ===")
    print(f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}")
    
    # Load data
    df = trainer.load_data(DATA_PATH, sample_size=SAMPLE_SIZE)
    if df is None:
        print("Failed to load dataset. Please check the file path.")
        return
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Train models
    results = trainer.train_models(X_train, X_val, y_train, y_val)
    
    if results is None:
        print("Training failed!")
        return
    
    # Plot model comparison
    trainer.plot_model_comparison(results)
    
    # Evaluate best model
    trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model(MODEL_SAVE_PATH)
    
    # Test predictions
    print("\n=== Testing Predictions ===")
    test_texts = [
        "I love this movie! It's absolutely fantastic!",
        "This is the worst experience ever. Terrible service.",
        "The weather is okay today, nothing special.",
        "Amazing product! Highly recommended!",
        "I'm so disappointed with this purchase.",
        "Great job everyone! Keep up the excellent work!",
        "This app crashes all the time. Very frustrating.",
        "Pretty good overall, could be better but decent.",
        "Absolutely horrible! Want my money back!",
        "Best purchase I've made in years!"
    ]
    
    predictions = trainer.predict_sentiment(test_texts)
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print(f"Sentiment: {pred['sentiment']}")
        if pred['confidence']:
            print(f"Confidence: {pred['confidence']:.4f}")
        print("-" * 60)
    
    return trainer

if __name__ == "__main__":
    print("Sentiment Analysis Model Training (No NLTK Dependencies)")
    print("=" * 60)
    
    # Run the training
    trainer = main()
