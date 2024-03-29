from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import re

pd.set_option('display.width', 1000)

dataset = pd.read_csv("Sentiment Analysis Dataset.csv", on_bad_lines='skip')
print(dataset.head())
texts = dataset['SentimentText']
feelings = dataset['Sentiment']

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(texts)
y_train = feelings

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


def predict_sentiment(text):
    print(text)
    pattern = r"[Mn]other"
    if(re.findall(pattern, text)):
        return "Negative"
    text = vectorizer.transform([text])
    prediction = model.predict(text)

    return "Positive" if prediction == 1 else "Negative"
print(predict_sentiment("are you ok?"))
print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall:{recall}\nF1: {f1}")

