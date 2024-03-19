import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))

path_train = os.path.join(dir_path, r"data/clean/RECCON_train.csv")
path_test = os.path.join(dir_path, r"data/clean/RECCON_test.csv")

ds = load_dataset("csv", data_files={"train": path_train, "test": path_test})

model_ckpt = "prajjwal1/bert-tiny"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = len(label2id)

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt, do_lower_case=True)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

dse = ds.map(tokenize, batched=True)

metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="micro")

batch_size = 16
logging_steps = len(dse["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=4,
    learning_rate=5e-5,
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

eval = trainer.evaluate(dse["test"])
print(eval)