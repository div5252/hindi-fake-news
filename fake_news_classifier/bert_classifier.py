import transformers
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import datasets
import pandas as pd
import torch
from transformers import DataCollatorWithPadding
import numpy as np
import argparse 
import os

parser = argparse.ArgumentParser(description='Bert Training Arguments')
parser.add_argument('--train_data', type = str, required=True)
parser.add_argument('--test_data', type = str, required=True)
parser.add_argument('--save_dir', type = str, required=True)
parser.add_argument('--num_epochs', type = int, required=True)


args = vars(parser.parse_args())

train_file  = os.path.abspath(args['train_data'])
test_file = os.path.abspath(args['test_data'])
save_dir = os.path.abspath(args['save_dir'])
epochs = args['num_epochs']

model_checkpoint = "bert-base-multilingual-cased"
batch_size = 1
seq_max_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_labels = 2


dataset = load_dataset('csv', data_files={'train': [train_file],
                                              'test': [test_file]})

print(dataset["train"][0])
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(example):
    return tokenizer(example["body"], example["heading"], truncation=True, padding=True, max_length=seq_max_length, add_special_tokens=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=2)
model.to(device)

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir = save_dir,
    learning_rate = 5e-6
)

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

validation_key = "test"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)


trainer.train()
