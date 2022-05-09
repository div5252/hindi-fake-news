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
1. Navigate to the fake_news_generation directory
 ```sh
cd fake_news_generator
```
2. Run the pos.py script
```sh
python pos.py --input_file=<input real news csv file> --output_file=<output file containing the results>
```

### Using POS replacement
1. Navigate to the fake_news_generation directory
 ```sh
cd fake_news_generator
```
2. Run the pos.py script
```sh
python pos.py --person_list=<csv containing hindi names> --location_list=<csv containing hindi locations> --organisation_list=<csv containing hindi organization names> --input_file=<input real news csv> --output_file=<output file name> --num_steps=<number of steps at which writing takes place>
``` 

## Fake New Classification
