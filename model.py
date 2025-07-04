import joblib
from preprocessing import wordopt
import pandas as pd

newsmodel = joblib.load("fake_newsmodel.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def manual_testing(news):
    testing_news = {"text": [news]}  # Create DataFrame from input text
    new_def_test = pd.DataFrame(testing_news)
    
    # Text preprocessing (assuming you have a wordopt() function for cleaning)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    
    # Vectorization (Assuming 'vectorizer' is your trained TfidfVectorizer)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    
    # Prediction using Logistic Regression
    pred_lr = newsmodel.predict(new_xv_test)

    # return "\n\nLR Prediction: {}".format(output_label(pred_lr[0]))
    return pred_lr[0]


def output_label(n):
    if n== 0:
        return "It is a Fake News."
    elif n== 1:
        return "It is a Genuine News."
    

# news_artical = input("Enter the Artical:")

# result = manual_testing(news_artical)
# print(result)