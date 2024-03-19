import os
import pandas as pd
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))

fn = ["test", "train", "val"]

test_text = []
test_labels = []

file_path = r"data\emotions-dataset-for-nlp\test.txt"
with open(os.path.join(dir_path, file_path), "r") as f:
    data = f.readlines()

for line in data:
    t, l = line.split(";")
    t, l = t.strip(), l.strip()
    test_text.append(t)
    test_labels.append(l)

train_text = []
train_labels = []

file_path = r"data\emotions-dataset-for-nlp\train.txt"
with open(os.path.join(dir_path, file_path), "r") as f:
    data = f.readlines()

for line in data:
    t, l = line.split(";")
    t, l = t.strip(), l.strip()
    train_text.append(t)
    train_labels.append(l)

file_path = r"data\emotions-dataset-for-nlp\val.txt"
with open(os.path.join(dir_path, file_path), "r") as f:
    data = f.readlines()

for line in data:
    t, l = line.split(";")
    t, l = t.strip(), l.strip()
    train_text.append(t)
    train_labels.append(l)

train_dict = {"text": train_text, "labels": train_labels}
test_dict = {"text": test_text, "labels": test_labels}

df_train = pd.DataFrame(train_dict)
df_test = pd.DataFrame(test_dict)

df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)

df_train["labels"].replace(["joy"], ["happiness"], inplace=True)
df_test["labels"].replace(["joy"], ["happiness"], inplace=True)

ft = ["love"]

for f in ft:
    df_train = df_train[df_train["labels"] != f]
    df_test = df_test[df_test["labels"] != f]

df_train["labels"].replace(label2id, inplace=True)
df_test["labels"].replace(label2id, inplace=True)

print(df_train["labels"].value_counts(ascending=True))
print(df_test["labels"].value_counts(ascending=True))

csv_train = df_train.to_csv(os.path.join(dir_path, r"data\clean\EDFnlp_train.csv"), index=False)
csv_test = df_test.to_csv(os.path.join(dir_path, r"data\clean\EDFnlp_test.csv"), index=False)