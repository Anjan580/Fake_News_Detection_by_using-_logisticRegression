import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from preprocessing import wordopt


true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

true['label'] = 1
fake['label'] = 0

news =pd.concat([fake, true], axis = 0)

news = news.drop(['title', 'subject', 'date'], axis = 1)

news = news.sample(frac = 1) # reshuffling

news.reset_index(inplace = True)

news.drop(['index'], axis = 1, inplace = True)

news['text'] = news['text'].apply(wordopt)

x = news['text']
y = news['label']

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.3, random_state= 7)

vectorization = TfidfVectorizer()

xv_train = vectorization.fit_transform(x_train)

xv_test = vectorization.transform(x_test)

# Logistic Regression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Save model & vectorizer

joblib.dump(LR, 'fake_newsmodel.pkl')
joblib.dump(vectorization, "vectorizer.pkl")
