#  Fake News Detection using Machine Learning

## Overview

The **Fake News Detection System** is a machine learning-based project
that classifies news articles as *Real* or *Fake* using text processing
and Logistic Regression.
This project demonstrates how Natural Language Processing (NLP) and
machine learning can be applied to tackle misinformation by analyzing
textual patterns and extracting features using TF-IDF.

------------------------------------------------------------------------

## Objectives

-   Detect and classify news as **Real** or **Fake**.
-   Preprocess text using **tokenization, stopword removal, cleaning,
    and TF-IDF vectorization**.
-   Train and evaluate a machine learning model (Logistic Regression).
-   Achieve a reliable accuracy score (\~96%) suitable for academic &
    real-world use.
-   Build a easy to use user interface(UI)

------------------------------------------------------------------------

## Key Results

-   **Train Accuracy:** \~96%
-   **Test Accuracy:** \~96%
-   **Model Used:** Logistic Regression
-   **Vectorizer:** TF-IDF
-   Balanced accuracy across training/testing proves stable performance.

------------------------------------------------------------------------

## Workflow Pipeline

### 1. **Data Preprocessing**

-   Lowercasing
-   Removing special characters & punctuations
-   Stopword removal
-   Lemmatization
-   Converting text to numerical vectors using **TF-IDF**

### 2. **Model Training**

-   Train/test split (70%--30%)
-   Model: **Logistic Regression**
-   Fitted using TF-IDF transformed text

### 3. **Evaluation**

-   Accuracy Score
-   Precision
-   Recall
-   F1-Score
-   Confusion Matrix
-   Classification Report
-   No overfitting (train ≈ test accuracy)

------------------------------------------------------------------------

## Tools & Technologies

-   **Python**
-   **Pandas**, **NumPy**
-   **Matplotlib**, **Seaborn**(For confusion matrix)
-   **Scikit-learn** (model, evaluation, TF-IDF)
-   **NLTK / re** (text preprocessing)
-   **Jupyter Notebook / VS Code**

------------------------------------------------------------------------

## Dataset Description

The dataset generally includes the following columns: - **title** --
headline of the news
- **text** -- full news article
- **subject** -- category/topic
- **date** -- publication date
- **label** -- *0 = Real*, *1 = Fake*

------------------------------------------------------------------------

## Features

-   Robust text cleaning pipeline
-   TF-IDF based feature extraction
-   High classification accuracy
-   Easy to deploy & extend
-   Useful for research and academic projects (CSIT final year)

------------------------------------------------------------------------

## Future Enhancements

-   Add **Deep Learning models** (LSTM, Bi-LSTM)
-   Use **BERT or other transformer models** for higher accuracy
-   Add real-time prediction API
-   Improve dataset size and balance

------------------------------------------------------------------------

## Author

**Anjani Kumar Pokhrel**
Kathmandu, Nepal
[LinkedIn](https://www.linkedin.com/in/anjan-pokhrel-441425283/)\
[GitHub](https://github.com/Anjan580)
