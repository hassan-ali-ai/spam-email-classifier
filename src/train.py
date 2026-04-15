import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
from preprocess import clean_text

# Sample dataset (you can replace later)
data = {
    "text": [
        "Win money now!!!",
        "Hello how are you",
        "Claim your prize now",
        "Let's meet tomorrow",
        "Congratulations you won lottery"
    ],
    "label": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Preprocess
df['text'] = df['text'].apply(clean_text)

# Features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../model/spam_model.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')

print("Model trained and saved successfully!")