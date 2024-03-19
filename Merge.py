import os
import pandas as pd
from sklearn.model_selection import train_test_split
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))
clean_path = os.path.join(dir_path, r"data/clean")

cleanfiles = os.listdir(clean_path)

df = pd.DataFrame({"text": [], "labels": []})

for file in cleanfiles:
    if "Merge" not in file:
        dff = pd.read_csv(os.path.join(clean_path, file))
        df = pd.concat((df, dff))

df.reset_index(inplace=True, drop=True)
counts = df["labels"].value_counts(ascending=True)
cmin = min(counts)
dfg = df.groupby("labels")
cat = []
dfg.apply(lambda x: cat.append(x.sample(n=cmin)))

df = pd.DataFrame({"text": [], "labels": []})

for dfc in cat:
    df = pd.concat((df, dfc))

df_train, df_test = train_test_split(df, stratify=df["labels"], test_size=0.1)
df_train.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)

print(df_train["labels"].value_counts(ascending=True))
print(df_test["labels"].value_counts(ascending=True))

csv_train = df_train.to_csv(os.path.join(dir_path, r"data\clean\Merge_train.csv"), index=False)
csv_test = df_test.to_csv(os.path.join(dir_path, r"data\clean\Merge_test.csv"), index=False)