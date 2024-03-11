import pandas as pd
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df_train = pd.read_csv(r"data/clean/RECCON_train.csv")
df_test = pd.read_csv(r"data/clean/RECCON_test.csv")

X_train = df_train["text"]
X_test = df_test["text"]
y_train = df_train["labels"].replace({
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise"
})
y_test = df_test["labels"].replace({
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise"
})

# Convert text data into numerical features using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Train the Decision Tree model
dt_model = naive_bayes.MultinomialNB()
dt_model.fit(X_train_tfidf, y_train)

# Evaluate the model on the test set
y_pred = dt_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

example_data = ["Stop playing your phone!"]
example_data_tfidf = tfidf_vectorizer.transform(example_data)
prediction = dt_model.predict(example_data_tfidf)
print("Predicted emotion:", prediction)