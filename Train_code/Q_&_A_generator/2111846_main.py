import time
import os
from datasets import load_dataset, load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder

# --------------- Load and Preprocess Dataset ---------------

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the original dataset (for demonstration purposes)
dataset = load_dataset("squad_v2")

# Reduce dataset size to 3%
train_dataset = dataset['train'].train_test_split(test_size=0.97, seed=42)['train']
val_dataset = dataset['validation'].train_test_split(test_size=0.97, seed=42)['train']

# Preprocessing function to tokenize and prepare inputs/labels
def preprocess_function(examples):
    inputs = ["Generate a question and answer from the context: " + context for context in examples['context']]
    targets = [
        f"Question: {question} Answer: {answer['text'][0] if answer['text'] else 'No answer'}"
        for question, answer in zip(examples['question'], examples['answers'])
    ]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to both train and validation sets
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

print(train_dataset)
print(val_dataset)

# Save the processed datasets to disk
save_dir = "./data/preprocessedDataset/processed_squad_v2"
os.makedirs(save_dir, exist_ok=True)
train_dataset.save_to_disk(os.path.join(save_dir, "train"))
val_dataset.save_to_disk(os.path.join(save_dir, "validation"))

# --------------- Load the Processed Dataset ---------------
# If already processed and saved, you can skip this part
train_dataset = load_from_disk("./data/preprocessedDataset/processed_squad_v2/train")
val_dataset = load_from_disk("./data/preprocessedDataset/processed_squad_v2/validation")

# Training arguments
training_args = TrainingArguments(
    output_dir="./training/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./training/logs',
    logging_steps=500,
    report_to="none",
    save_total_limit=1,
    save_steps=1000,
    save_strategy="steps",
    run_name="T5_Squad_Training_Run"
)

# Load model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Metrics storage
metrics = {}

# Start training and measure time
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
metrics["training_time"] = training_time

# Save the trained model
model.save_pretrained("./training/model/t5_squad_model")
tokenizer.save_pretrained("./training/model/t5_squad_model")

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
with open('eval_metrics.txt', 'w') as f:
  for metric, value in metrics.items():
    f.write(f"{metric}: {value}\n")
