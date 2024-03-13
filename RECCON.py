import os
import json
import pandas as pd
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))

file_path = 'data\RECCON-main\data\original_annotation\dailydialog_train.json'
with open(os.path.join(dir_path, file_path), "r") as json_file:
    data = json.load(json_file)

# Extract features (utterances) and labels (emotions)
X_train = []
y_train = []
for key, conversations in data.items():
    for conversation in conversations:
        for utterance in conversation:
            X_train.append(utterance['utterance'])
            y_train.append(utterance['emotion'])

file_path = r'data\RECCON-main\data\subtask2\fold1\dailydialog_classification_train_without_context.csv'
data = pd.read_csv(os.path.join(dir_path, file_path))

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_train.append(utterance)
        y_train.append(emotions)

file_path = r'data\RECCON-main\data\subtask2\fold1\dailydialog_classification_train_with_context.csv'
data = pd.read_csv(os.path.join(dir_path, file_path))

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_train.append(utterance)
        y_train.append(emotions)

file_path = 'data\RECCON-main\data\original_annotation\dailydialog_test.json'
with open(os.path.join(dir_path, file_path), "r") as json_file:
    data = json.load(json_file)

X_test = []
y_test = []
for key, conversations in data.items():
    for conversation in conversations:
        for utterance in conversation:
            X_test.append(utterance['utterance'])
            y_test.append(utterance['emotion'])

file_path = r'data\RECCON-main\data\subtask2\fold1\dailydialog_classification_test_without_context.csv'
data = pd.read_csv(os.path.join(dir_path, file_path))

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_test.append(utterance)
        y_test.append(emotions)

file_path = r'data\RECCON-main\data\subtask2\fold1\dailydialog_classification_test_with_context.csv'
data = pd.read_csv(os.path.join(dir_path, file_path))

s = data['text'].tolist()
for item in s:
    emotions, utterances = item.split(" <SEP> ", 1)
    utterance_list = utterances.split(" <SEP> ")
    for utterance in utterance_list:
        X_test.append(utterance)
        y_test.append(emotions)

train_dict = {"text": X_train, "labels": y_train}
test_dict = {"text": X_test, "labels": y_test}

df_train = pd.DataFrame(train_dict)
df_test = pd.DataFrame(test_dict)

df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)

df_train["labels"].replace(["sad", "happy", "surprised", "angry"], ["sadness", "happiness", "surprise", "anger"], inplace=True)
df_test["labels"].replace(["sad", "happy", "surprised", "angry", "happines"], ["sadness", "happiness", "surprise", "anger", "happiness"], inplace=True)

df_train = df_train[df_train["labels"] != "excited"]
df_test = df_test[df_test["labels"] != "excited"]

# print(df_train["labels"].value_counts(ascending=True))
# print(df_test["labels"].value_counts(ascending=True))

df_train["labels"].replace(label2id, inplace=True)
df_test["labels"].replace(label2id, inplace=True)

print(df_train["labels"].value_counts(ascending=True))
print(df_test["labels"].value_counts(ascending=True))

csv_train = df_train.to_csv(os.path.join(dir_path, r"data\clean\RECCON_train.csv"), index=False)
csv_test = df_test.to_csv(os.path.join(dir_path, r"data\clean\RECCON_test.csv"), index=False)