import os
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report

script_dir = os.path.dirname(__file__)
file_path = 'RECCON-main\data\original_annotation\dailydialog_train.json'
file_path = os.path.join(script_dir, file_path)
with open(file_path, "r") as json_file:
    data = json.load(json_file)

# Extract features (utterances) and labels (emotions)
X_train = []
y_train = []
for key, conversations in data.items():
    for conversation in conversations:
        for utterance in conversation:
            X_train.append(utterance['utterance'])
            y_train.append(utterance['emotion'])

file_path = r'RECCON-main\data\subtask2\fold1\dailydialog_classification_train_without_context.csv'
file_path = os.path.join(script_dir, file_path)
data = pd.read_csv(file_path)

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_train.append(utterance)
        y_train.append(emotions)

file_path = r'RECCON-main\data\subtask2\fold1\dailydialog_classification_train_with_context.csv'
file_path = os.path.join(script_dir, file_path)
data = pd.read_csv(file_path)

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_train.append(utterance)
        y_train.append(emotions)

# file_path = r'RECCON-main\data\subtask2\fold2\dailydialog_classification_train_without_context.csv'
# file_path = os.path.join(script_dir, file_path)
# data = pd.read_csv(file_path)

# s = data['text'].tolist()
# for item in s:
#     emotions, utterances = item.split(" <SEP> ", 1)
#     utterance_list = utterances.split(" <SEP> ")
#     for utterance in utterance_list:
#         X_train.append(utterance)
#         y_train.append(emotions)

# file_path = r'RECCON-main\data\subtask2\fold2\dailydialog_classification_train_with_context.csv'
# file_path = os.path.join(script_dir, file_path)
# data = pd.read_csv(file_path)

# s = data['text'].tolist()
# for item in s:
#     emotions, utterances = item.split(" <SEP> ", 1)
#     utterance_list = utterances.split(" <SEP> ")
#     for utterance in utterance_list:
#         X_train.append(utterance)
#         y_train.append(emotions)

file_path = 'RECCON-main\data\original_annotation\dailydialog_test.json'
file_path = os.path.join(script_dir, file_path)
with open(file_path, "r") as json_file:
    data = json.load(json_file)

X_test = []
y_test = []
for key, conversations in data.items():
    for conversation in conversations:
        for utterance in conversation:
            X_test.append(utterance['utterance'])
            y_test.append(utterance['emotion'])

file_path = r'RECCON-main\data\subtask2\fold1\dailydialog_classification_test_without_context.csv'
file_path = os.path.join(script_dir, file_path)
data = pd.read_csv(file_path)

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_test.append(utterance)
        y_test.append(emotions)

file_path = r'RECCON-main\data\subtask2\fold1\dailydialog_classification_test_with_context.csv'
file_path = os.path.join(script_dir, file_path)
data = pd.read_csv(file_path)

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_test.append(utterance)
        y_test.append(emotions)

# file_path = r'RECCON-main\data\subtask2\fold2\dailydialog_classification_test_without_context.csv'
# file_path = os.path.join(script_dir, file_path)
# data = pd.read_csv(file_path)

# s = data['text'].tolist()
# for item in s:
#     emotions, utterances = item.split(" <SEP> ", 1)
#     utterance_list = utterances.split(" <SEP> ")
#     for utterance in utterance_list:
#         X_test.append(utterance)
#         y_test.append(emotions)

# file_path = r'RECCON-main\data\subtask2\fold2\dailydialog_classification_test_with_context.csv'
# file_path = os.path.join(script_dir, file_path)
# data = pd.read_csv(file_path)

# s = data['text'].tolist()
# for item in s:
#     emotions, utterances = item.split(" <SEP> ", 1)
#     utterance_list = utterances.split(" <SEP> ")
#     for utterance in utterance_list:
#         X_test.append(utterance)
#         y_test.append(emotions)

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

