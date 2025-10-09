# train.py
import argparse
import numpy as np
import os
from datasets import load_dataset, load_metric
from transformers import (
AutoConfig,
AutoTokenizer,
AutoModelForTokenClassification,
DataCollatorForTokenClassification,
TrainingArguments,
Trainer,
)
import torch




def parse_args():
parser = argparse.ArgumentParser(description="Train NER model (token classification)")
parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
parser.add_argument("--output_dir", type=str, default="models/ner-model")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
return parser.parse_args()




def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True, max_length=128):
# `examples['tokens']` is a list of token lists
tokenized_inputs = tokenizer(
examples["tokens"],
is_split_into_words=True,
truncation=True,
max_length=max_length,
)


labels = []
for i, label in enumerate(examples["ner_tags"]):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx = None
    
label_ids = []
for word_idx in word_ids:
	if word_idx is None:
	label_ids.append(-100)
	elif word_idx != previous_word_idx:
	# label for first token of the word
	label_ids.append(label[word_idx])
	else:
# For the other tokens in a word
	label_ids.append(label[word_idx] if label_all_tokens else -100)
	previous_word_idx = word_idx
	labels.append(label_ids)


	tokenized_inputs["labels"] = labels
	return tokenized_inputs




def main():
args = parse_args()


# reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
main()