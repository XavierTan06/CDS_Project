import os
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))

df_train = pd.read_csv(os.path.join(dir_path, r"data/clean/EDFnlp_train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, r"data/clean/EDFnlp_test.csv"))

X_train = df_train["text"]
X_test = df_test["text"]
y_train = df_train["labels"].replace(id2label)
y_test = df_test["labels"].replace(id2label)

# Convert text data into numerical features using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Train the Decision Tree model
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)

# Evaluate the model on the test set
y_pred = dt_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
f1 = f1_score(y_test, y_pred, average="micro")
print(f"Overall f1: {f1}")

example_data = ["Stop playing your phone!"]
example_data_tfidf = tfidf_vectorizer.transform(example_data)
prediction = dt_model.predict(example_data_tfidf)
print("Predicted emotion:", prediction)