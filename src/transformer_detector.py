from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import json

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def get_data(data_path, random_seed, out_domain='bulgarian'):
    """
    function to read dataframe with columns
    """
    en_sources = ['wikihow', 'wikipedia', 'reddit', 'arxiv', 'peerread', 'outfox']
    
    data = pd.read_json(data_path, lines=True)
    data['source'] = [source if source not in en_sources else 'english' for source in data['source']]
    
    if out_domain != 'no':
        test_df = data[data['source'] == out_domain]
        train_df = data[data['source'] != out_domain]
    else:
        train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=random_seed)
    
    train_df['id'] = train_df.index
    test_df['id'] = test_df.index
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)
    
    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id    # put your model here
    )
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)


def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds


if __name__ == '__main__':

    data_path = 'SubtaskB_split.jsonl'
    data = pd.read_json(data_path, lines=True)
    random_seeds = [0, 10, 30, 55, 75]
    model = "roberta-base" # or 'xlm-roberta-base'
    
    subtask =  'B' 
    multilingual = False

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}

        if multilingual:
            domains = ['german', 'english', 'bulgarian', 'urdu', 'italian', 'no', 'indonesian', 'chinese', 'russian', 'arabic']
        else:
            domains = list(data.model.unique())
            domains.append('no') # train on all and test on all generators
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly', 6: 'gpt4'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5, 'gpt4': 6}
    
        domains = list(data.source.unique())
        domains.append('no') # train on all and test on all domains

    for random_seed in random_seeds:
        for domain in domains:
            set_seed(random_seed)
            
            if multilingual:
                train_df, valid_df, test_df = get_data(data_path, random_seed, domain)
            else:
                train_df = data[data[f'{domain}_{random_seed}'] == 'train']
                valid_df = data[data[f'{domain}_{random_seed}'] == 'valid']
                test_df = data[data[f'{domain}_{random_seed}'] == 'test']

            fine_tune(train_df, valid_df, f"{model}_/subtask{subtask}/{random_seed}", id2label, label2id, model)
            results, predictions = test(test_df, f"{model}_/subtask{subtask}/{random_seed}/best/", id2label, label2id)
            
            with open(f'classification_report_results_{data_path.split(".")[0]}.jsonl', 'a') as f: 
                results_data = {
                    'subtask': subtask,
                    'experiment': domain,
                    'random_seed': random_seed,
                    'test_set': domain,
                    'model': model,
                    'results': results
                }
                f.write(json.dumps(results_data) + '\n')
            
