import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='NER fake news Arguments')
parser.add_argument('--person_list', type=str, required=True)
parser.add_argument('--location_list', type=str, required=True)
parser.add_argument('--organisation_list', type=str, required=True)
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--type_dataset', type=str, required=True)
parser.add_argument('--num_steps', type=int, required=True)

args = vars(parser.parse_args())

person_list = os.path.abspath(args['person_list'])
location_list = os.path.abspath(args['location_list'])
organisation_list = os.path.abspath(args['organisation_list'])
input_file = os.path.abspath(args['input_file'])
output_file = os.path.abspath(args['output_file'])
type_dataset = args['type_dataset']
num_steps = args['num_steps']


def ner(example):
    ner_results = nlp(example)
    ner_results_grouped = []
    curr_word = ''
    curr_token = ''
    curr_start = -1
    curr_end = -1
    prev_token = 0
    prev_end = -2
    entity_map = {'PER': 1, 'LOC': 2, 'ORG': 3}
    for r in ner_results:
        if prev_token == 0 or (entity_map[r['entity']] == prev_token and r['start'] - prev_end <= 1):
            if prev_token == 0:
                curr_start = r['start']
            curr_word += r['word']
            curr_end = r['end']
            curr_token = r['entity']
        else:
            if curr_word.startswith('▁'):
                curr_word = curr_word[1:]
            curr_word = curr_word.replace('▁', ' ')
            ner_results_grouped.append(
                {'entity': curr_token, 'start': curr_start, 'end': curr_end, 'word': curr_word})
            curr_start = r['start']
            curr_word = r['word']
            curr_end = r['end']
            curr_token = r['entity']
        prev_token = entity_map[r['entity']]
        prev_end = r['end']
    if curr_start != -1:
        if curr_word.startswith('▁'):
            curr_word = curr_word[1:]
        curr_word = curr_word.replace('▁', ' ')
        ner_results_grouped.append(
            {'entity': curr_token, 'start': curr_start, 'end': curr_end, 'word': curr_word})
    return ner_results_grouped


def ner_replace(article, ner_list, per_list, loc_list, org_list):
    replaced_article = article
    for ele in ner_list:
        if ele['word'].count(' ') > 2:
            continue
        if ele['entity'] == 'PER':
            index = random.randrange(len(per_list))
            replaced_article = replaced_article.replace(
                ele['word'], per_list[index])
        if ele['entity'] == 'LOC':
            index = random.randrange(len(loc_list))
            replaced_article = replaced_article.replace(
                ele['word'], loc_list[index])
        if ele['entity'] == 'ORG':
            index = random.randrange(len(org_list))
            replaced_article = replaced_article.replace(
                ele['word'], org_list[index])
    return replaced_article


def start_of_article(article):
    article = str(article)
    article += " "
    for i in range(min(511, len(article) - 1), 0, -1):
        if article[i] == ' ':
            return article[0:i], article[i:]
    return "", article


def ner_replace_complete(article, per_list, loc_list, org_list):
    splits = []
    if type_dataset == 'bbc':
        splits = article.split('.')
    else:
        splits = article.split('।')

    ner_list = []
    for i in range(len(splits)):
        start, _ = start_of_article(splits[i])
        ner_list += ner(start)

    replaced = ner_replace(article, ner_list, per_list, loc_list, org_list)
    return replaced


tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-r-ner-40-lang")
model = AutoModelForTokenClassification.from_pretrained(
    "jplu/tf-xlm-r-ner-40-lang", from_tf=True)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

per_df = pd.read_csv(person_list)
loc_df = pd.read_csv(location_list)
org_df = pd.read_csv(organisation_list)
per_list = per_df['name']
loc_list = loc_df['name']
org_list = org_df['name']


dataset = pd.read_csv(input_file, encoding='utf-8')

fake_dataset = dataset
for i in tqdm(range(len(dataset['body']))):
    fake_dataset['body'][i] = ner_replace_complete(
        dataset['body'][i], per_list, loc_list, org_list)
    fake_dataset['label'][i] = 1
    if i % num_steps == (num_steps - 1):
        fake_dataset_i = fake_dataset[0:i]
        fake_dataset_i.to_csv(output_file, index=False)

fake_dataset.to_csv(output_file, index=False)
