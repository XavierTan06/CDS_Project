import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

model_ckpt = "/saved/Merge_TinyBert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    while True:
        print("Enter sample sentence for inference:")
        text = input()
        inference = clf(text)
        print(inference)