from flask import Flask, render_template, request
import joblib
import os
import sys

# Fix import path
sys.path.append('src')
from preprocess import clean_text

app = Flask(__name__)

# Load model
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]

    if result == 1:
        prediction = "Spam ❌"
    else:
        prediction = "Not Spam ✅"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)