import pandas as pd
from labelMap import label2id, id2label
from sklearn.model_selection import train_test_split

text = []
labels = []

df = pd.read_csv(r"data\TweetEmotions\tweet_emotions.csv")
df.drop("tweet_id", axis=1, inplace=True)
df.rename(columns={"sentiment": "labels", "content": "text"}, inplace=True)
df.drop_duplicates(inplace=True)

df = df[df["labels"] != "fun"]
df = df[df["labels"] != "hate"]
df = df[df["labels"] != "enthusiasm"]
df = df[df["labels"] != "empty"]
df = df[df["labels"] != "boredom"]

df["labels"].replace(label2id, inplace=True)

df_train, df_test = train_test_split(df)

print(df_train["labels"].value_counts(ascending=True))
print(df_test["labels"].value_counts(ascending=True))

csv_train = df_train.to_csv(r"data\clean\TweetEmotions_train.csv", index=False)
csv_test = df_test.to_csv(r"data\clean\TweetEmotions_test.csv", index=False)
