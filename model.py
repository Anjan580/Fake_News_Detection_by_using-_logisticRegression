import joblib
from preprocessing import wordopt
import pandas as pd
from Alogrithm import LogisticRegressionSparse

newsmodel = joblib.load("LR_model.pkl")
vectorizer = joblib.load("vectorization.pkl")

def manual_testing(news):
    testing_news = {"text": [news]}  # Create DataFrame from input text
    new_def_test = pd.DataFrame(testing_news)
    
    # Text preprocessing (assuming you have a wordopt() function for cleaning)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    
    # Vectorization (Assuming 'vectorizer' is your trained TfidfVectorizer)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    
    # Prediction using Logistic Regression
    pred_proba = newsmodel.predict_proba(new_xv_test)[0]  # probabilities [p_fake, p_real]
    pred_class = newsmodel.predict(new_xv_test)[0]

    return pred_class, pred_proba


def output_label(n):
    if n== 0:
        return "This might be Fake. "
    elif n== 1:
        return "This might be Genuine News."
    
