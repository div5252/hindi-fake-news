import nltk
import pandas as pd
import stanza
nltk.download('wordnet')
from nltk.corpus import wordnet
from tqdm import tqdm
import torch
import argparse 
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

parser = argparse.ArgumentParser(description='POS fake data generation arguments')
parser.add_argument('--input_file', type = str, required=True)
parser.add_argument('--output_file', type = str, required=True)



args = vars(parser.parse_args())

input_file  = os.path.abspath(args['input_file'])
output_file = os.path.abspath(args['output_file'])


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def start_of_article(article):
  article = str(article)
  article += " "
  for i in range(min(511, len(article) - 1), 0, -1):
    if article[i] == ' ':
      break
  return article[0:i], article[i:]

def antonym(word):
  antonyms = []
  for syn in wordnet.synsets(word):
    for l in syn.lemmas():
      if l.antonyms():
        antonyms.append(l.antonyms()[0].name())
  if len(antonyms) == 0:
    return -1
  return antonyms[0]

stanza.download('hi',  package = 'hdtb')
hi_tagger = stanza.Pipeline(lang='hi', processors='tokenize,pos',package = 'hdtb')

stanza.download('en',  package = 'partut')
en_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos',package = 'partut')

def translate(text):
    tokenizer.src_lang = "en_XX"
    encoded_text = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def backtranslate(text):
    tokenizer.src_lang = "hi_IN"
    encoded_text = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def tag_replace(text):
    en_text = backtranslate(text)
    doc = en_tagger(en_text)
    replaced_text = text
    for k in doc.to_dict():
        for t in k:
          if 'upos' in t.keys() and 'text' in t.keys():
            if t['upos'] == "ADJ" or t['upos'] == "VERB":
                item = t["text"]
                if antonym(item) == -1:
                    pass
                else:
                    anto = antonym(item)
                    en_text = en_text.replace(item,anto)
    replaced_text = translate(en_text)
    return replaced_text

def pos_replacer(article):
    start,end = start_of_article(article)
    replaced = tag_replace(start)
    return replaced + end

dataset = pd.read_csv(input_file, encoding='utf-8')

fake_dataset = dataset
for i in tqdm(range(29000, len(dataset['body']))):
  start_part, end_part = start_of_article(dataset['body'][i])
  fake_dataset['body'][i] = pos_replacer(start_part) + end_part
  fake_dataset['label'][i] = 1
  if i % 100 == 99:
    fake_dataset.to_csv(output_file, index=False)
    
fake_dataset.to_csv(output_file, index=False)
