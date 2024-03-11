import os
import json
import pandas as pd

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

test_dict = {"text": X_test, "emotion": y_test}
train_dict = {"text": X_train, "emotion": y_train}

df_test = pd.DataFrame(test_dict)
df_train = pd.DataFrame(train_dict)

csv_test = df_test.to_csv(os.path.join(script_dir, r"RECCON-main\data\transform\test.csv"), index=False)
csv_train = df_train.to_csv(os.path.join(script_dir, r"RECCON-main\data\transform\train.csv"), index=False)