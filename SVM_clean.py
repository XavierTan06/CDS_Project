import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report

script_dir = os.path.dirname(__file__)

df_train = pd.read_csv(os.path.join(script_dir, r"RECCON-main/data/transform/train.csv"))
df_test = pd.read_csv(os.path.join(script_dir, r"RECCON-main/data/transform/test.csv"))

X_train = df_train["text"]
X_test = df_test["text"]
y_train = df_train["labels"]
y_test = df_test["labels"]

# Convert text data into numerical features using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Normalize numerical features
x_scaler = MaxAbsScaler()
x_scaler.fit(X_train_tfidf)
X_train_norm = x_scaler.transform(X_train_tfidf)
X_test_norm = x_scaler.transform(X_test_tfidf)

# Train the SVM model
svm_model = SVC(kernel='sigmoid')
# {'poly', 'sigmoid', 'linear', 'precomputed', 'rbf'}
svm_model.fit(X_train_norm, y_train)

# Evaluate the model on the test set
y_pred = svm_model.predict(X_test_norm)
print(classification_report(y_test, y_pred))

example_data = ["Stop playing your phone!"]
example_data_tfidf = tfidf_vectorizer.transform(example_data)
example_data_norm = x_scaler.transform(example_data_tfidf)
prediction = svm_model.predict(example_data_norm)
print("Predicted emotion:", prediction)