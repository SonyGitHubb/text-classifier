
import os
import glob
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Zařízení a batch size
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 if torch.cuda.is_available() else 4

# Načtení CSV souborů z datasetu
files = glob.glob("data/*.csv")
frames = []
for f in files:
    label = os.path.splitext(os.path.basename(f))[0].capitalize()
    df = pd.read_csv(f)
    df["label"] = label
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# Sloučení sloupců title + perex do jednoho textu
df["text"] = df["title"].astype(str) + " " + df["perex"].astype(str)
df["label_id"] = df["label"].astype("category").cat.codes
labels = list(df["label"].astype("category").cat.categories)

# Rozdělení datasetu
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label_id"], random_state=42)

# Tokenizace
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

def to_dataset(dataframe):
    ds = Dataset.from_pandas(dataframe).map(
        tokenize, batched=True, remove_columns=["title", "perex", "text", "label"]
    )
    ds = ds.rename_column("label_id", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

train_ds = to_dataset(train_df)
val_ds = to_dataset(val_df)
test_ds = to_dataset(test_df)

# Model 
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(labels),
    id2label={i: label for i, label in enumerate(labels)},
    label2id={label: i for i, label in enumerate(labels)}
).to(device)

# Metody vyhodnocení 
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(pred.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# Trénování 
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

# Vyhodnocení
metrics = trainer.evaluate(test_ds)
print(f"Accuracy: {metrics['eval_accuracy']:.3f}  F1: {metrics['eval_f1']:.3f}")

preds = trainer.predict(test_ds)
print(classification_report(test_df["label_id"], preds.predictions.argmax(-1), target_names=labels, digits=3))

# Uložení modelu 
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")