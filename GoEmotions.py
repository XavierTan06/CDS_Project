import os
import pandas as pd
import numpy as np
from labelMap import label2id, id2label
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.realpath(__file__))

text = []
labels = []

df = pd.read_parquet(os.path.join(dir_path, r"data\GoEmotions\train-00000-of-00001.parquet"))

# df.rename(columns={"joy": "happiness"})

inc = set(label2id.keys()) & set(df.columns.to_list())

for label in df.columns.to_list():
    # print(label)
    if label not in inc and label != "text":
        df.drop(label, axis=1, inplace=True)
print(df.head())
# df = df[df.sum(axis=1) == 1]
df = df[(df.iloc[:, 1:] == 1).any(axis=1)]
for label in inc:
    df.loc[df[label]==1, "labels"] = label

df = df[["text", "labels"]]
df.reset_index(inplace=True, drop=True)

df["labels"].replace(label2id, inplace=True)
# df["labels"] = df["labels"].astype(int)

df_train, df_test = train_test_split(df)

print(df_train["labels"].value_counts(ascending=True))
print(df_test["labels"].value_counts(ascending=True))

csv_train = df_train.to_csv(os.path.join(dir_path, r"data\clean\GoEmotions_train.csv"), index=False)
csv_test = df_test.to_csv(os.path.join(dir_path, r"data\clean\GoEmotions_test.csv"), index=False)
