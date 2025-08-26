import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

true = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

true['label'] = 1
fake['label'] = 0

news =pd.concat([fake, true], axis = 0)

news = news.drop(['title', 'subject', 'date'], axis = 1)
news = news.sample(frac = 1)
news.reset_index(inplace = True)
news.drop(['index'], axis = 1, inplace = True)
X = news['text']
y = news['label']


# implement Logistic regression mannually


from scipy.sparse import csr_matrix

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sparse Logistic Regression
class LogisticRegressionSparse:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):
            # Sparse matrix multiplication
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)

            # Gradient calculation
            error = y_pred - y
            dw = (X.T.dot(error)) / m   # works with sparse
            db = np.sum(error) / m

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return np.where(y_pred > 0.5, 1, 0)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

# Vectorization
vectorization = TfidfVectorizer(max_features=10000)

xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
joblib.dump(vectorization, 'vectorization.pkl')

# Train the model
model = LogisticRegressionSparse(lr=0.1, epochs=1000)
model.fit(xv_train, y_train)

# Predictions
y_pred_train = model.predict(xv_train)
y_pred_test = model.predict(xv_test)

# Accuracy
accuracy = lambda y_true, y_pred: np.mean(y_true == y_pred)
print("Train Accuracy:", accuracy(y_train, y_pred_train))
print("Test Accuracy:", accuracy(y_test, y_pred_test))

# Save the model
joblib.dump(model, 'LR_model.pkl')
