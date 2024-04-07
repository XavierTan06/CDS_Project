import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from labelMap import id2label
import numpy as np

#Load data
dir_path = os.path.dirname(os.path.realpath(__file__))

df_train = pd.read_csv(os.path.join(dir_path, r"data\TextEmotion\TextEmotion_train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, r"data\TextEmotion\TextEmotion_test.csv"))

X_train = df_train["content"]
X_test = df_test["content"]
y_train = df_train["sentiment"]
y_test = df_test["sentiment"]

# Replace NaN values with empty strings. At least for this dataset, this step can't be done before splitting the data.
X_train = X_train.fillna('')
X_test = X_test.fillna('')

# Tokenize text as n-grams and convert them into vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Build and run MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=100, verbose=True)
mlp_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = mlp_model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Overall f1: {f1}")