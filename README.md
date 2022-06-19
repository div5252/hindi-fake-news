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
## News articles scrapping
1. Navigate to the news_scrapper directory
 ```sh
cd news_scrapper
```
2. Make a new directory data where downloaded news article will be stored
```sh
mkdir data
```
3. Run the navbharat_scrapper.py script
```sh
python navbharat_scrapper.py
``` 

## Fake News Generation

Link to the dataset - https://drive.google.com/drive/folders/1YoEf0FxC_TNIgVlakFE3HjT0LK9EIkv2?usp=sharing

### Using Split and Merge

1. Navigate to the fake_news_generator directory
 ```sh
cd fake_news_generator
```
2. Run the split_and_merge.py script
```sh
python split_and_merge.py
``` 
 
### Using NER replacement
1. Navigate to the fake_news_generator directory
 ```sh
cd fake_news_generator
```
2. Run the .py script
```sh
python ner.py --person_list=<csv containing hindi names> --location_list=<csv containing hindi locations> --organisation_list=<csv containing hindi organization names> --input_file=<input real news csv> --output_file=<output file name> --num_steps=<number of steps at which writing takes place>
```

### Using POS replacement
1. Navigate to the fake_news_generator directory
 ```sh
cd fake_news_generator
```
2. Run the pos.py script
```sh
python pos.py --input_file=<input real news csv file> --output_file=<output file containing the results>
```

## Fake News Classification

### Classification using similarity features
1. Navigate to the fake_news_classifier directory
 ```sh
cd fake_news_classifier
```
2. Run the news_similarity_classifier.py file
 ```sh
python news_similarity_classifier.py --train_path=<train dataset csv> --dev_path=<dev dataset csv> --test_path=<test dataset csv> --gold_path=<gold dataset csv> --result_path=<output text file> [--use_sentiment_features=<whether use sentiment features or not>] [--bert_dir=<bert model file for similarity features>] 
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
