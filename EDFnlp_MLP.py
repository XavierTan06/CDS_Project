import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from labelMap import id2label

#Load data
dir_path = os.path.dirname(os.path.realpath(__file__))

trainset = pd.read_csv(os.path.join(dir_path, r"data/clean/EDFnlp_train.csv"))
testset = pd.read_csv(os.path.join(dir_path, r"data/clean/EDFnlp_test.csv"))

trainsetX = trainset["text"]
testsetX = testset["text"]
trainsetY = trainset["labels"].replace(id2label)
testsetY = testset["labels"].replace(id2label)

#Tokenize text as n-grams and convert them into vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
trainVecX = vectorizer.fit_transform(trainsetX)
testVecX = vectorizer.transform(testsetX)

#Build and run MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=100, verbose=True)
mlp_model.fit(trainVecX, trainsetY)

#Evaluate
y_pred = mlp_model.predict(testVecX)
print("Classification Report:")
print(classification_report(testsetY, y_pred))
f1 = f1_score(y_test, y_pred, average="micro")
print(f"Overall f1: {f1}")