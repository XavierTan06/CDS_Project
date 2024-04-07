import os
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))
rawfile_path = os.path.join(dir_path, r"data\\TextEmotion\\text_emotion.csv") # Change as needed 

df = pd.read_csv(rawfile_path)
df.drop(["tweet_id", "author"], axis=1, inplace=True)
df=df[df["sentiment"].isin(label2id.keys())]

""" def remove_punctuations(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text """

def remove(text):
    text = re.sub(r"(@\S+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r"\d+\S*|\S*\d+", "", text)
    return text

df['content'] = df['content'].str.lower()
df['content'] = df['content'].apply(remove)
stop = stopwords.words('english')
df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(df)

df_train, df_test = train_test_split(df, stratify=df["sentiment"], test_size=0.1)
df_train.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)

print(df_train["sentiment"].value_counts(ascending=True))
print(df_test["sentiment"].value_counts(ascending=True))

csv_train = df_train.to_csv(os.path.join(dir_path, r"data\TextEmotion\TextEmotion_train.csv"), index=False)
csv_test = df_test.to_csv(os.path.join(dir_path, r"data\TextEmotion\TextEmotion_test.csv"), index=False) 