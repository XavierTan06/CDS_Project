import os
import torch
from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from labelMap import id2label, label2id

dir_path = os.path.dirname(os.path.realpath(__file__))
model_ckpt = "./saved/GoEmotions_TinyBert"

path_test = os.path.join(dir_path, r"data/clean/GoEmotions_test.csv")

ds = load_dataset("csv", data_files={"test": path_test})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
infer = clf(ds["test"]["text"])
y_pred = [i["label"] for i in infer]
y_test_id = ds["test"]["labels"]
y_test = [id2label[y] for y in y_test_id]

classification_rep = classification_report(y_test, y_pred)
print(classification_rep)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Overall f1: {f1}")

with open("output/GoEmotions - GoEmotions/GoEmotions_TinyBert.txt", "w") as file:
    file.write(classification_rep)
    file.write("\n")
    file.write("Overall f1-score: " + str(f1))