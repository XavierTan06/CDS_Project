import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

script_dir = os.path.dirname(__file__)

path_train = os.path.join(script_dir, r"RECCON-main/data/transform/train.csv")
path_test = os.path.join(script_dir, r"RECCON-main/data/transform/test.csv")

ds = load_dataset("csv", data_files={"train": path_train, "test": path_test})

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dse = ds.map(tokenize, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = 7
id2label = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}
label2id = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id).to(device)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

batch_size = 32
logging_steps = len(dse["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="debug"
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dse["train"],
    eval_dataset=dse["test"],
    tokenizer=tokenizer
)

trainer.train()

pred_output = trainer.predict(dse["test"])
pred_output.metrics