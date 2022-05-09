# Hindi fake News Generation and Classification

## Setting up the Repository
1. Clone the repository
```sh
git clone git@github.com:div5252/hindi-fake-news.git
```
2. Install the required dependencies in your python environment
```sh
pip install requirements.txt
```
## Fake News Generation

### Using NER replacement
1. Navigate to the fake_news_generator directory
 ```sh
cd fake_news_generator
```
2. Run the pos.py script
```sh
python pos.py --input_file=<input real news csv file> --output_file=<output file containing the results>
```

### Using POS replacement
1. Navigate to the fake_news_generator directory
 ```sh
cd fake_news_generator
```
2. Run the ner.py script
```sh
python ner.py --person_list=<csv containing hindi names> --location_list=<csv containing hindi locations> --organisation_list=<csv containing hindi organization names> --input_file=<input real news csv> --output_file=<output file name> --num_steps=<number of steps at which writing takes place>
``` 

## Fake New Classification

### Classification using similarity features
1. Navigate to the fake_news_classifier directory
 ```sh
cd fake_news_classifier
```
2. Run the news_similarity_classifier.py file
 ```sh
python news_similarity_classifier.py --bert_dir=<bert model file for similarity features> --train_file=<train dataset csv> --test_file=<test dataset csv> 
```

### Classification using BERT
1. Navigate to the fake_news_classifier directory
 ```sh
cd fake_news_classifier
```
2. Run the bert_classifier.py file
 ```sh
python bert_classifier.py --train_data=<train dataset csv> --test_data=<test dataset csv> --save_dir=<output directory> --num_epochs=<number of epochs>
```
