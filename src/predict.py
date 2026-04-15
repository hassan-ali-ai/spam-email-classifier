import joblib
from preprocess import clean_text

model = joblib.load('../model/spam_model.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

def predict_email(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    
    if result == 1:
        return "Spam"
    else:
        return "Not Spam"

# Test
if __name__ == "__main__":
    email = input("Enter email text: ")
    print(predict_email(email))