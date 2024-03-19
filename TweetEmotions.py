import os
import pandas as pd
from labelMap import label2id, id2label
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.realpath(__file__))

text = []
labels = []

df = pd.read_csv(os.path.join(dir_path, r"data\TweetEmotions\tweet_emotions.csv"))
df.drop("tweet_id", axis=1, inplace=True)
df.rename(columns={"sentiment": "labels", "content": "text"}, inplace=True)
df.drop_duplicates(inplace=True)

ft = ["fun", "hate", "enthusiasm", "empty", "boredom", "love", "worry", "relief"]

for f in ft:
    df = df[df["labels"] != f]

df["labels"].replace(label2id, inplace=True)

df_train, df_test = train_test_split(df)

print(df_train["labels"].value_counts(ascending=True))
print(df_test["labels"].value_counts(ascending=True))

csv_train = df_train.to_csv(os.path.join(dir_path, r"data\clean\TweetEmotions_train.csv"), index=False)
csv_test = df_test.to_csv(os.path.join(dir_path, r"data\clean\TweetEmotions_test.csv"), index=False)
