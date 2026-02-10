# ==============================
# SENTIMENT ANALYSIS PROJECT
# End-to-End Single File Script
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import pickle

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. Load Dataset
# Make sure "IMDB Dataset.csv" is in the same folder
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())


# 3. Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)        # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove numbers & symbols
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)


# 4. Encode Target Variable
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})


# 5. Train-Test Split
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 7. Model Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# 8. Model Evaluation
y_pred = model.predict(X_test_tfidf)

print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 9. Prediction Function
def predict_sentiment(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    
    if prediction[0] == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"


# 10. Test Prediction
print("\nSample Predictions")
print(predict_sentiment("This movie was fantastic and inspiring"))
print(predict_sentiment("Worst movie, completely waste of time"))


# 11. Save Model and Vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and Vectorizer Saved Successfully")