#pip install datasets
#pip install transformers

#pip install -U negate
#from negate import Negator

'''
# Use default model (en_core_web_md):
negator = Negator()

# Or use a Transformer model (en_core_web_trf):
negator = Negator(use_transformers=True)

sentence = "An apple a day, keeps the doctor away."
sentence = "PersonX holds PersonY's attention"
sentence = "gets fresh air"
sentence = "PersonX puts this ___ into practice"
#sentence = "fun"
#negated_sentence = negator.negate_sentence(sentence)
#print(negated_sentence)  # "An apple a day, doesn't keep the doctor away."
'''

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
#link1 = root+'/atomic_train_data/1_all_training_data(pos,neg,annotated).csv'
link2 = root+'/2_test_data.csv'
link_test_all = root+'/2_test_data_1488.csv'
link3 = root+'/1_annotated_data.csv'
#link4 = root+'/1_atomic.csv'
link4 = root+'/1_atomic_42k_1.csv'
#link5 = root+'/1_anion_logical_neg.csv'
link5 = root+'/1_logical_neg_1500_p_32k_1.csv'
#link6 = root+'/1_anion_semi_logical_neg.csv'
link6 = root+'/1_semi_logical_neg_1500_p_33k_1.csv'

test_data = pd.read_csv(link2)
test_data_all = pd.read_csv(link_test_all)
annotated_data = pd.read_csv(link3)
atomic_data = pd.read_csv(link4)
anion_logical_neg_data_label_1 = pd.read_csv(link5)
anion_semi_logical_neg_data_label_1 = pd.read_csv(link6)

with open('output.txt', 'w') as file:
  print(len(test_data), file = file)
  print(len(annotated_data), file = file)
  print(len(test_data_all), file = file)
  print(len(atomic_data), file = file)
  print(len(anion_logical_neg_data_label_1), file = file)
  print(len(anion_semi_logical_neg_data_label_1), file = file)

#for now
#link7 = root+'/2_atomic_data_minus.csv'
link7 = root+'/1_atomic_60k_0.csv'
#link8 = root+'/2_anion_logical_neg_data_minus.csv'
link8 = root+'/1_logical_neg_1500_p_60k_0.csv'
#link9 = root+'/2_anion_semi_logical_neg_data_minus.csv'
link9 = root+'/1_semi_logical_neg_1500_p_60k_0.csv'

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
  if size < 5000:
    t_args = TrainingArguments(
      output_dir='./results',          # output directory
      per_device_train_batch_size=16,  # batch size per device during training
      per_device_eval_batch_size=20,   # batch size for evaluation
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
      # but you can specify `metric_for_best_model` argument to change to accuracy or other metric

      num_train_epochs=8,              # total number of training epochs
      warmup_steps=50,                # number of warmup steps for learning rate scheduler
      logging_steps=50,               # log & save weights each logging_steps
      save_steps=50,

      learning_rate=lr,
      seed=42,
      evaluation_strategy="steps",     # evaluate each `logging_steps`
    )
  elif size < 10000:
    t_args = TrainingArguments(
      output_dir='./results',          # output directory
      per_device_train_batch_size=16,  # batch size per device during training
      per_device_eval_batch_size=20,   # batch size for evaluation
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
      # but you can specify `metric_for_best_model` argument to change to accuracy or other metric

      num_train_epochs=2.5,              # total number of training epochs
      warmup_steps=100,                # number of warmup steps for learning rate scheduler
      logging_steps=100,               # log & save weights each logging_steps
      save_steps=100,

      learning_rate=lr,
      seed=42,
      evaluation_strategy="steps",     # evaluate each `logging_steps`
    )
  else:
    t_args = TrainingArguments(
      output_dir='./results',          # output directory
      per_device_train_batch_size=16,  # batch size per device during training
      per_device_eval_batch_size=20,   # batch size for evaluation
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
      # but you can specify `metric_for_best_model` argument to change to accuracy or other metric

      num_train_epochs=2,              # total number of training epochs
      warmup_steps=100,                # number of warmup steps for learning rate scheduler
      logging_steps=100,               # log & save weights each logging_steps
      save_steps=100,

      learning_rate=lr,
      seed=42,
      evaluation_strategy="steps",     # evaluate each `logging_steps`
    )
  return t_args

import json
import csv
import random

def extract_combination(data):
  all_rows = []
  total_row = len(data) - 1
  def get_random_number(present_row):
    while True:
      random_number = random.randint(0, total_row)
      rand_data = data[random_number]
      if (abs(random_number - present_row) != 0):
        break
    return random_number

  for row_number, item in enumerate(data):
    random_num = get_random_number(row_number)

    #print(f"Row number: {row_number}, Data: {item}")
    #print('rand data: ', data[random_num])

    p = item['p']
    r = item['r']
    q2 = data[random_num]['q']

    no_p = negator.negate_sentence(p)
    no_q2 = negator.negate_sentence(q2)
    #print('no p: ', no_p)
    #print('q2: ', q2)
    #print('no q2: ', no_q2)

    w_count = len(no_q2.split())
    if (w_count == 1):
      no_q2 = 'not ' + no_q2

    if (no_q2.startswith("To don't") or no_q2.startswith("to don't")):
      words = no_q2.split()
      no_q2 = " ".join(words[1:])

    #print('new no q2: ', no_q2)

    #find no_p
    #find no_q2

    # create 3 new data then add to dataset
    # (p, r, q2), (p, r, no_q2), (no_p, r, q2) with label 0
    # what about (no_p, r, no_q2)
    tuple_1 = [p, q2, r, 0]
    tuple_2 = [p, no_q2, r, 0]
    tuple_3 = [no_p, q2, r, 0]

    all_rows.append(tuple_1)
    all_rows.append(tuple_2)
    all_rows.append(tuple_3)
  return all_rows

train_size = 30000
for i in range(27):
  # read the csv file
  annotated_data = pd.read_csv(link3)
  atomic_data = pd.read_csv(link4)
  anion_logical_neg_data_label_1 = pd.read_csv(link5)
  anion_semi_logical_neg_data_label_1 = pd.read_csv(link6)

  # generate first, upload and then use
  atomic_data_minus = pd.read_csv(link7)
  anion_logical_neg_data_minus = pd.read_csv(link8)
  anion_semi_logical_neg_data_minus = pd.read_csv(link9)

  if i == 0:
    #continue
    with open('output.txt', 'a') as file:
      print("Dataset: annotated data", file = file)

    # process annotated data
    train_data = annotated_data
  elif i == 1:
    continue
    # process annotated data + ATOMIC(+)
    atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    atomic_data = atomic_data.head(train_size) # For now, train with only 5000

    train_data = pd.concat([annotated_data, atomic_data], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + atomic(+)", file = file)
  elif i == 2:
    continue
    # process annotated data + ANION_Logical_Neg(+)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    train_data = pd.concat([annotated_data, anion_logical_neg_data_label_1], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + anion_logical_neg(+)", file = file)
  elif i == 3:
    continue
    # process annotated data + ANION_Semi_Logical_Neg(+)
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    train_data = pd.concat([annotated_data, anion_semi_logical_neg_data_label_1], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Semi_Logical_Neg(+)", file = file)
  elif i == 4:
    continue
    # process annotated data + ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    train_data = pd.concat([annotated_data, anion_logical_neg_data_label_1, anion_semi_logical_neg_data_label_1], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + anion_logical_neg(+) + anion_semi_logical_neg(+)", file = file)
  elif i == 5:
    continue
      '''
      continue
      atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      atomic_data = atomic_data.head(5000) # For now, train with only 5000

      # process annotated data + ATOMIC(-)
      atomic_data = atomic_data.reset_index(drop=True)
      atomic_data = Dataset.from_pandas(atomic_data)

      atomic_data_minus = extract_combination(atomic_data)

      # Specify column names
      column_names = ['p', 'q', 'r', 'output']
      # Create a DataFrame
      atomic_data_minus = pd.DataFrame(atomic_data_minus, columns=column_names)
      atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility

      # Specify the file path where you want to save the CSV file
      file_path = 'atomic_data_minus.csv'

      # Save the DataFrame to a CSV file
      atomic_data_minus.to_csv(file_path, index=False)
      '''

    atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
    atomic_data_minus = atomic_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, atomic_data_minus], axis=0) #axis = 0 means row wise concatanation
    
    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ATOMIC(-)", file = file)
  elif i == 6:
    continue
      '''
      continue
      # process annotated data + ANION_Logical_Neg(-)
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(5000) # For now, train with only 5000

      anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.reset_index(drop=True)
      anion_logical_neg_data_label_1 = Dataset.from_pandas(anion_logical_neg_data_label_1)

      anion_logical_neg_data_minus = extract_combination(anion_logical_neg_data_label_1)

      # Specify column names
      column_names = ['p', 'q', 'r', 'output']
      # Create a DataFrame
      anion_logical_neg_data_minus = pd.DataFrame(anion_logical_neg_data_minus, columns=column_names)
      anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility

      # Specify the file path where you want to save the CSV file
      file_path = 'anion_logical_neg_data_minus.csv'

      # Save the DataFrame to a CSV file
      anion_logical_neg_data_minus.to_csv(file_path, index=False)
      '''

    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, anion_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Logical_Neg(-)", file = file)
  elif i == 7:
    continue
    '''
    #continue
    # process annotated data + ANION_Semi_Logical_Neg(-)
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(5000) # For now, train with only 5000

    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.reset_index(drop=True)
    anion_semi_logical_neg_data_label_1 = Dataset.from_pandas(anion_semi_logical_neg_data_label_1)

    anion_semi_logical_neg_data_minus = extract_combination(anion_semi_logical_neg_data_label_1)

    # Specify column names
    column_names = ['p', 'q', 'r', 'output']
    # Create a DataFrame
    anion_semi_logical_neg_data_minus = pd.DataFrame(anion_semi_logical_neg_data_minus, columns=column_names)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility

    # Specify the file path where you want to save the CSV file
    file_path = 'anion_semi_logical_neg_data_minus.csv'

    # Save the DataFrame to a CSV file
    anion_semi_logical_neg_data_minus.to_csv(file_path, index=False)
    '''

    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 8:
    continue
    # process annotated data + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, anion_logical_neg_data_minus, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 9:
    continue
    # process annotated data + ATOMIC(+) + ATOMIC(-)
    atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    atomic_data = atomic_data.head(train_size) # For now, train with only 5000

    atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
    atomic_data_minus = atomic_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, atomic_data, atomic_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated_data + ATOMIC(+) + ATOMIC(-)", file = file)
  elif i == 10:
    continue
    # process annotated data + ANION_Logical_Neg(+) + ANION_Logical_Neg(-)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, anion_logical_neg_data_label_1, anion_logical_neg_data_minus], axis=0)

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Logical_Neg(+) + ANION_Logical_Neg(-)", file = file)
  elif i == 11:
    continue
    # process annotated data + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 12:
    continue
    # process annotated data + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([annotated_data, anion_logical_neg_data_label_1, anion_logical_neg_data_minus,
                            anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0)

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)", file = file)  
  elif i == 13:
    continue
    # process annotated data + ATOMIC (+) + ATOMIC (-) +
    # ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)

    atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    atomic_data = atomic_data.head(train_size) # For now, train with only 5000

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

    train_data = pd.concat([annotated_data, atomic_data, atomic_data_minus,
                            anion_logical_neg_data_label_1, anion_logical_neg_data_minus,
                            anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0)

    with open('output.txt', 'a') as file:
      print("Dataset: annotated data + ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 14:
    continue
    # process ATOMIC(+)
    atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    atomic_data = atomic_data.head(train_size) # For now, train with only 5000

    train_data = atomic_data

    with open('output.txt', 'a') as file:
      print("Dataset: ATOMIC(+)", file = file)
  elif i == 15:
    continue
    # process ANION_Logical_Neg(+)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    train_data = anion_logical_neg_data_label_1

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Logical_Neg(+)", file = file)
  elif i == 16:
    continue
    # process ANION_Semi_Logical_Neg(+)
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    train_data = anion_semi_logical_neg_data_label_1

    with open('output.txt', 'a') as file:
      print("Dataset: anion_semi_logical_neg(+)", file = file)
  elif i == 17:
    continue
    # process ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    train_data = pd.concat([anion_logical_neg_data_label_1, anion_semi_logical_neg_data_label_1], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+)", file = file)
  elif i == 18:
    continue
    atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
    atomic_data_minus = atomic_data_minus.head(train_size)

    train_data = atomic_data_minus

    with open('output.txt', 'a') as file:
      print("Dataset: ATOMIC(-)", file = file)
  elif i == 19:
    continue
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    train_data = anion_logical_neg_data_minus

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Logical_Neg(-)", file = file)
  elif i == 20:
    continue
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = anion_semi_logical_neg_data_minus

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 21:
    continue
    # process ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([anion_logical_neg_data_minus, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 22:
    continue
    # process ATOMIC(+) + ATOMIC(-)
    atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    atomic_data = atomic_data.head(train_size) # For now, train with only 5000
    #print(atomic_data.head(15))

    atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
    atomic_data_minus = atomic_data_minus.head(train_size)
    #print(atomic_data_minus.head(15))

    train_data = pd.concat([atomic_data, atomic_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: ATOMIC(+) + ATOMIC(-)", file = file)
  elif i == 23:
    continue
    # process ANION_Logical_Neg(+) + ANION_Logical_Neg(-)
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([anion_logical_neg_data_label_1, anion_logical_neg_data_minus], axis=0)

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Logical_Neg(+) + ANION_Logical_Neg(-)", file = file)
  elif i == 24:
    continue
    # process ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) # For now, train with only 5000

    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
    anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

    train_data = pd.concat([anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

    with open('output.txt', 'a') as file:
      print("Dataset: ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)", file = file)
  elif i == 25:
    continue
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
      print("Dataset: ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)", file = file)  
  elif i == 26:
    continue
    # process ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
    atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
    atomic_data = atomic_data.head(train_size) # For now, train with only 5000

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
      print("Dataset: ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)", file = file)

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

  new_test_dataset = test_dataset
  if '__index_level_0__' in test_dataset.column_names:
    new_test_dataset = test_dataset.remove_columns(['__index_level_0__'])
  new_test_dataset

  dts = Dataset.from_pandas(new_train_dataset.to_pandas()).train_test_split(test_size=0.10)
  print(dts)

  dataset = DatasetDict()

  dataset['train'] = Dataset.from_pandas(dts["train"].to_pandas())
  dataset['validation'] = Dataset.from_pandas(dts["test"].to_pandas())
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
  os.system("git add .")
  os.system("git commit -m message")
  os.system("git push")

os.system("git add .")
os.system("git commit -m message")
os.system("git push")
