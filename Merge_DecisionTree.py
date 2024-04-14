import os
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))

df_train = pd.read_csv(os.path.join(dir_path, r"data/clean/Merge_train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, r"data/clean/Merge_test.csv"))

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
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from labelMap import id2label

#Load data
dir_path = os.path.dirname(os.path.realpath(__file__))

df_train = pd.read_csv(os.path.join(dir_path, r"data/clean/GoEmotions_train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, r"data/clean/GoEmotions_test.csv"))

X_train = df_train["text"]
X_test = df_test["text"]
y_train = df_train["labels"].replace(id2label)
y_test = df_test["labels"].replace(id2label)

#Tokenize text as n-grams and convert them into vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Build and run MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=100, verbose=True)
mlp_model.fit(X_train_tfidf, y_train)

#Evaluate
y_pred = mlp_model.predict(X_test_tfidf)
classification_rep = classification_report(y_test, y_pred)
print(classification_rep)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Overall f1: {f1}")

with open("output/Merge_DecisionTree.txt", "w") as file:
    file.write(classification_rep)
    file.write("\n")
    file.write("Overall f1-score: " + str(f1))

example_data = ["I can’t go in there, it’s too dark and I keep hearing strange noises!"]
example_data_tfidf = tfidf_vectorizer.transform(example_data)
prediction = dt_model.predict(example_data_tfidf)
print("Predicted emotion:", prediction)