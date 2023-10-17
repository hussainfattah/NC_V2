#pip install datasets
#pip install transformers
#pip install scikit-learn
#pip install scipy
#pip install torch

#pip install -U negate
#from negate import Negator

from datasets import Dataset, DatasetDict
import pandas as pd
from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import TrainingArguments, Trainer
from scipy.special import softmax
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback
import json
import csv
import random
import os 

import torch
torch.cuda.empty_cache()

root="../data"
link2 = root+'/2_test_data.csv'
link_test_all = root+'/2_test_data_1488.csv'
link3 = root+'/1_annotated_data.csv'

link4 = root+'/1_atomic_all_p.csv'
link5 = root+'/1_logical_neg_all_p.csv'
link6 = root+'/1_semi_logical_neg_all_p.csv'

#test_data = pd.read_csv(link2)
test_data_all = pd.read_csv(link_test_all)
#annotated_data = pd.read_csv(link3)
atomic_data = pd.read_csv(link4)
anion_logical_neg_data_label_1 = pd.read_csv(link5)
anion_semi_logical_neg_data_label_1 = pd.read_csv(link6)

with open('output.txt', 'w') as file:
  #print(len(test_data), file = file)
  #print(len(annotated_data), file = file)
  print(len(test_data_all), file = file)
  print(len(atomic_data), file = file)
  print(len(anion_logical_neg_data_label_1), file = file)
  print(len(anion_semi_logical_neg_data_label_1), file = file)

link7 = root+'/0_atomic_all.csv'
link8 = root+'/0_logical_neg_all.csv'
link9 = root+'/0_semi_logical_neg_all.csv'

atomic_data_minus = pd.read_csv(link7)
anion_logical_neg_data_minus = pd.read_csv(link8)
anion_semi_logical_neg_data_minus = pd.read_csv(link9)

with open('output.txt', 'a') as file:
  print(len(atomic_data_minus), file = file)
  print(len(anion_logical_neg_data_minus), file = file)
  print(len(anion_semi_logical_neg_data_minus), file = file)

#concat all and make a new string
def concat_all_by_sep_train(example):
  output = int(example['output'])

  final_str = example['p'] + " </s> " + example['r'] + " </s> " + example['q']

  return {'label': output, 'text': final_str}

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def custom_metrics_all(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    #micro and macro
    precision = metric1.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="micro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

#pip install accelerate==0.20.3

lr = 2e-5

def getTrainingArguments(size):
  epochs = 0
  step = 0
  if size < 5100:
    epochs = 8
    step = 50
  elif size < 10000:
    epochs = 2.5
    step = 100
  else:
    epochs = 2
    step = 100

  t_args = TrainingArguments(
    output_dir='./results',          # output directory
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric

    num_train_epochs=epochs,              # total number of training epochs
    warmup_steps=step,                # number of warmup steps for learning rate scheduler
    logging_steps=step,               # log & save weights each logging_steps
    save_steps=step,

    learning_rate=lr,
    seed=42,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
  )
  return t_args

size_list = [5000, 25000, 50000, 100000]
for train_size in size_list:
  # positive samples
  atomic_data = pd.read_csv(link4)
  anion_logical_neg_data_label_1 = pd.read_csv(link5)
  anion_semi_logical_neg_data_label_1 = pd.read_csv(link6)

  # negative samples
  atomic_data_minus = pd.read_csv(link7)
  anion_logical_neg_data_minus = pd.read_csv(link8)
  anion_semi_logical_neg_data_minus = pd.read_csv(link9)

  for i in range(13):
    if i == 0:
      # process ATOMIC(+)
      atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      atomic_data = atomic_data.head(train_size) 
      
      train_data = atomic_data
      
      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC(+), Size: ", train_size, file = file)
    elif i == 1:
      # process ANION_Semi_Logical_Neg(+)
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) 
      
      train_data = anion_logical_neg_data_label_1
      
      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Semi_Logical_Neg(+), Size: ", train_size, file = file)
    elif i == 2:
      # process ANION_Semi_Logical_Neg(+)
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      train_data = anion_semi_logical_neg_data_label_1

      with open('output.txt', 'a') as file:
        print("Dataset: anion_semi_logical_neg(+), Size: ", train_size, file = file)
    elif i == 3:
      # process ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+)
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      train_data = pd.concat([anion_logical_neg_data_label_1, anion_semi_logical_neg_data_label_1], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+), Size: ", train_size, file = file)
    elif i == 4:
      # process ATOMIC(-)
      atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
      atomic_data_minus = atomic_data_minus.head(train_size)
      train_data = atomic_data_minus

      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC(-), Size: ", train_size, file = file)
    elif i == 5:
      # logical neg (-)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)
      train_data = anion_logical_neg_data_minus

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 6:
      # semi logical neg (-)
      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

      train_data = anion_semi_logical_neg_data_minus

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 7:
      # process ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

      train_data = pd.concat([anion_logical_neg_data_minus, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 8:
      # process ATOMIC(+) + ATOMIC(-)
      atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      atomic_data = atomic_data.head(train_size)

      atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
      atomic_data_minus = atomic_data_minus.head(train_size)

      train_data = pd.concat([atomic_data, atomic_data_minus], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC(+) + ATOMIC(-), Size: ", train_size, file = file)
    elif i == 9:
      # process ANION_Logical_Neg(+) + ANION_Logical_Neg(-)
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

      train_data = pd.concat([anion_logical_neg_data_label_1, anion_logical_neg_data_minus], axis=0)

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(+) + ANION_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 10:
      # process ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

      train_data = pd.concat([anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 11:
      # process ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

      train_data = pd.concat([anion_logical_neg_data_label_1, anion_logical_neg_data_minus,
                            anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0)

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 12:
      # process ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
      atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      atomic_data = atomic_data.head(train_size) 

      atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
      atomic_data_minus = atomic_data_minus.head(train_size)

      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
      anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

      train_data = pd.concat([atomic_data, atomic_data_minus, anion_logical_neg_data_label_1, anion_logical_neg_data_minus,
                            anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0)

      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    
    print('Train data size: ', len(train_data))


'''
  print('Train data size: ', len(train_data))
  
  td = Dataset.from_pandas(train_data)
  if '__index_level_0__' in td.column_names:
    td = td.remove_columns(['__index_level_0__'])
  # Filter out rows where 'q' column has value 'nan'
  filtered_dataset = td.filter(lambda example: example['q'] != 'nan')
  # Filter out rows where 'q' attribute has value 'None'
  filtered_dataset = filtered_dataset.filter(lambda example: example['q'] is not None)

  train_dataset = Dataset.from_pandas(filtered_dataset.to_pandas())
  train_dataset = train_dataset.map(concat_all_by_sep_train)

  new_train_dataset = train_dataset.remove_columns(['p', 'q', 'r', 'output'])
  new_train_dataset
  new_train_dataset = new_train_dataset.shuffle(seed=42)

  test_dataset = Dataset.from_pandas(test_data)
  test_dataset = test_dataset.map(concat_all_by_sep_train)
  test_dataset

  test_dataset_all = Dataset.from_pandas(test_data_all)
  test_dataset_all = test_dataset_all.map(concat_all_by_sep_train)
  test_dataset_all

  new_test_dataset = test_dataset
  if '__index_level_0__' in test_dataset.column_names:
    new_test_dataset = test_dataset.remove_columns(['__index_level_0__'])
  new_test_dataset

  new_test_dataset_2 = test_dataset_all
  if '__index_level_0__' in test_dataset_all.column_names:
    new_test_dataset_2 = test_dataset_all.remove_columns(['__index_level_0__'])
  new_test_dataset_2

  dts = Dataset.from_pandas(new_train_dataset.to_pandas()).train_test_split(test_size=0.10)
  #print(dts)

  dataset = DatasetDict()

  dataset['train'] = Dataset.from_pandas(dts["train"].to_pandas())
  dataset['validation'] = Dataset.from_pandas(dts["test"].to_pandas())
  #dataset['test'] =  Dataset.from_pandas(new_test_dataset.to_pandas())

  if test_with_all == True:
    dataset['test'] =  Dataset.from_pandas(new_test_dataset_2.to_pandas())
  else:
    dataset['test'] =  Dataset.from_pandas(new_test_dataset.to_pandas())

  print(dataset)

  tokenized_datasets = dataset.map(tokenize_function, batched=True)

  checkpoint = "roberta-base"
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

  small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
  #small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(500))
  small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
  #print(small_train_dataset)
  #print(small_eval_dataset)

  lr = 2e-5

  training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=8,
                                  per_gpu_train_batch_size=16,
                                  seed = 123,
                                  learning_rate=lr)

  tr_args = getTrainingArguments(len(small_train_dataset))

  early_stop = EarlyStoppingCallback(3, 0.01)

  trainer = Trainer(
      model=model,
      args=tr_args,
      #train_dataset=tokenized_datasets["train"],
      train_dataset=small_train_dataset,
      eval_dataset=tokenized_datasets["validation"],
      #eval_dataset=small_eval_dataset,
      compute_metrics=custom_metrics_all,
      callbacks=[early_stop])

  trainer.train()
  trainer.evaluate()

  t = tokenized_datasets["test"].remove_columns("text")
  results = trainer.predict(t)
  results

  preds = []

  for x in results[0]:
    y = np.argmax(x)
    preds.append(y)
  set(preds)
  actual = results[1].tolist()

  with open('output.txt', 'a') as file:
    print(classification_report(actual, preds), file = file)
  #os.system("git add .")
  #os.system("git commit -m message")
  #os.system("git push")
'''

os.system("git add .")
os.system("git commit -m message")
os.system("git push")
