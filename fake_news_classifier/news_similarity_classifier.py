import pandas as pd
import numpy as np
import regex
from simpletransformers.classification import ClassificationModel
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm
import argparse
import os

def combine_head_body(df):
  def combine(row):
    return str(row['body']) + ' ' + str(row['heading'])

  df['combined'] = df.apply(combine, axis=1)


def hindi_analyzer(text):
  words = regex.findall(r'\w{2,}', text)
  for i in range(len(words)):
    yield words[i]
  for i in range(len(words) - 1):
    yield words[i] + " " + words[i + 1]
  for i in range(len(words) - 2):
    yield words[i] + " " + words[i + 1] + " " + words[i + 2]


class SentimentFeatures():
  def model(self, bert_dir):
    bert = ClassificationModel('bert', bert_dir, use_cuda=torch.cuda.is_available(), args={
        'reprocess_input_data': True,
        'use_cached_eval_features': False,
        'overwrite_output_dir': True,
        'silent': True
    })
    return bert

  def similarity(self, head, body):
    ans = np.abs(np.array(head) - np.array(body))
    return ans 


class CountFeatures():
  def fit(self, texts):
    count_vectorizer = CountVectorizer(analyzer = hindi_analyzer)
    count_fit = count_vectorizer.fit(texts)
    return count_fit

  def count_vec(self, fit, texts):
    count_transform = fit.transform(texts)
    return count_transform.toarray()
  
  def bool_vec(self, count_array):
    for i in range(len(count_array)):
      for j in range(len(count_array[i])):
        if count_array[i][j] > 0:
          count_array[i][j] = 1
    return count_array
  
  def overlap(self, head, body):
    bool_head = self.bool_vec(head)
    bool_body = self.bool_vec(body)
    ans = np.zeros((len(head)))
    for i in range(len(bool_head)):
      ans[i] = np.dot(bool_head[i], bool_body[i])
    return ans

  def normalized_overlap(self, head, body):
    ans = np.zeros((len(head)))
    for i in range(len(head)):
      for j in range(len(head[i])):
        if head[i][j] == 0:
          ans[i] += 0 
        else:
          ans[i] += float(body[i][j]) / float(head[i][j])
    return ans


class TfidfFeatures():
  def fit(self, texts):
    tfidf_vectorizer = TfidfVectorizer(analyzer = hindi_analyzer)
    tfidf_fit = tfidf_vectorizer.fit(texts)
    return tfidf_fit

  def tfidf_vec(self, fit, texts):
    tfidf_transform = fit.transform(texts)
    return tfidf_transform.toarray()

  def cosine_similarity(self, head, body):
    dot = np.zeros((len(head)))
    for i in range(len(head)):
      dot[i] = np.dot(head[i], body[i]) / (norm(head[i]) * norm(body[i]))
    return dot


class SvdFeatures():
  def fit(self, X):
    svd_vectorizer = TruncatedSVD(n_components=50)
    svd_fit = svd_vectorizer.fit(X)
    return svd_fit

  def svd_vec(self, fit, X):
    svd_transform = fit.transform(X)
    return svd_transform

  def cosine_similarity(self, head, body):
    dot = np.zeros((len(head)))
    for i in range(len(head)):
      dot[i] = np.dot(head[i], body[i]) / (norm(head[i]) * norm(body[i]))
    return dot


# Classification using Decision Tree
def dt_classifier(X_train, y_train, X_test, y_test):
  dt = DecisionTreeClassifier()
  dt.fit(X_train, y_train)
  dt_predicted = dt.predict(X_test)
  print("Accuracy using Decision Tree is", accuracy_score(y_test, dt_predicted), "\n")
  print("F1-score using Decision Tree is", f1_score(y_test, dt_predicted, average=None), "\n")


# Classification using SVM
def svm_classifier(X_train, y_train, X_test, y_test):
  svm = LinearSVC(max_iter=10000)
  svm.fit(X_train, y_train)
  svm_predicted = svm.predict(X_test)
  print("Accuracy using SVM is", accuracy_score(y_test, svm_predicted), "\n") 
  print("F1-score using SVM is", f1_score(y_test, svm_predicted, average=None), "\n") 


# Classification using AdaBoost
def adaboost_classifier(X_train, y_train, X_test, y_test):
  ada = AdaBoostClassifier()
  ada.fit(X_train, y_train)
  ada_predicted = ada.predict(X_test)
  print("Accuracy using AdaBoost is", accuracy_score(y_test, ada_predicted), "\n")
  print("F1-score using AdaBoost is", f1_score(y_test, ada_predicted, average=None), "\n")

# Classification using bagging
def bagging_classifier(X_train, y_train, X_test, y_test):
  bag = BaggingClassifier()
  bag.fit(X_train, y_train)
  bag_predicted = bag.predict(X_test)
  print("Accuracy using Bagging is", accuracy_score(y_test, bag_predicted), "\n")
  print("F1-score using Bagging is", f1_score(y_test, bag_predicted, average=None), "\n")


def xgboost_classifier(X_train, y_train, X_test, y_test):
  D_train = xgb.DMatrix(X_train, label=y_train)
  D_test = xgb.DMatrix(X_test, label=y_test)

  param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 2} 

  steps = 20

  model = xgb.train(param, D_train, steps)

  preds = model.predict(D_test)
  best_preds = np.asarray([np.argmax(line) for line in preds])

  print("Accuracy using xgb is", accuracy_score(y_test, best_preds), "\n")
  print("F1-score using xgb is", f1_score(y_test, best_preds, average=None), "\n")


# Main code
parser = argparse.ArgumentParser(description='News similarity classifier Arguments')
parser.add_argument('--bert_dir', type = str, required=True)
parser.add_argument('--train_file', type = str, required=True)
parser.add_argument('--test_file', type = str, required=True)

args = vars(parser.parse_args())

bert_dir  = os.path.abspath(args['bert_dir'])
train_file = os.path.abspath(args['train_file'])
test_file = os.path.abspath(args['test_file'])


train_path = train_file
test_path = test_file

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

n_train = train.shape[0]
n_test = test.shape[0]
    
combine_head_body(train)
combine_head_body(test)

train_sentiment = np.zeros((len(train),1))
train_overlap = np.zeros((len(train),1))
train_normalized_overlap = np.zeros((len(train),1))
train_tfidf_similarity = np.zeros((len(train),1))

for i in tqdm(range(len(train))):
  article = train[i:i+1]

  sentiment = SentimentFeatures()
  bert = sentiment.model(bert_dir)
  sentiment_train_head, _ = bert.predict(list(article['heading']))
  sentiment_train_body, _ = bert.predict(list(article['body']))

  train_sentiment[i] = sentiment.similarity(sentiment_train_head, sentiment_train_body)
      
  cf = CountFeatures()
  fit = cf.fit(article['combined'])
  count_train_head = cf.count_vec(fit, article['heading'])
  count_train_body = cf.count_vec(fit, article['body'])

  train_overlap[i] = cf.overlap(count_train_head, count_train_body)
  train_normalized_overlap[i] = cf.normalized_overlap(count_train_head, count_train_body)


  tf = TfidfFeatures()
  fit = tf.fit(article['combined'])
  tfidf_combined = tf.tfidf_vec(fit, article['combined'])
  tfidf_train_head = tf.tfidf_vec(fit, article['heading'])
  tfidf_train_body = tf.tfidf_vec(fit, article['body'])

  train_tfidf_similarity[i] = tf.cosine_similarity(tfidf_train_head, tfidf_train_body)

test_sentiment = np.zeros((len(train),1))
test_overlap = np.zeros((len(train),1))
test_normalized_overlap = np.zeros((len(train),1))
test_tfidf_similarity = np.zeros((len(train),1))

for i in tqdm(range(len(test))):
  article = test[i:i+1]

  sentiment = SentimentFeatures(bert_dir)
  bert = sentiment.model(bert_dir)
  sentiment_test_head, _ = bert.predict(list(article['heading']))
  sentiment_test_body, _ = bert.predict(list(article['body']))

  test_sentiment[i] = sentiment.similarity(sentiment_test_head, sentiment_test_body)

      
  cf = CountFeatures()
  fit = cf.fit(article['combined'])
  count_test_head = cf.count_vec(fit, article['heading'])
  count_test_body = cf.count_vec(fit, article['body'])

  test_overlap[i] = cf.overlap(count_test_head, count_test_body)
  test_normalized_overlap[i] = cf.normalized_overlap(count_test_head, count_test_body)


  tf = TfidfFeatures()
  fit = tf.fit(article['combined'])
  tfidf_combined = tf.tfidf_vec(fit, article['combined'])
  tfidf_test_head = tf.tfidf_vec(fit, article['heading'])
  tfidf_test_body = tf.tfidf_vec(fit, article['body'])

  test_tfidf_similarity[i] = tf.cosine_similarity(tfidf_test_head, tfidf_test_body)


train_combined_features = np.concatenate((train_sentiment, train_overlap, train_normalized_overlap, train_tfidf_similarity), axis=1)
test_combined_features = np.concatenate((test_sentiment, test_overlap, test_normalized_overlap, test_tfidf_similarity), axis=1)

print("\n\nDecision Tree -")
print("Using Sentiment features overlap -")
dt_classifier(train_sentiment, train['label'], test_sentiment, test['label'])
print("Using Count features overlap -")
dt_classifier(train_overlap, train['label'], test_overlap, test['label'])
print("Using Count features normalized overlap -")
dt_classifier(train_normalized_overlap, train['label'], test_normalized_overlap, test['label'])
print("Using Tfidf features -")
dt_classifier(train_tfidf_similarity, train['label'], test_tfidf_similarity, test['label'])
print("Combining all features - ")
dt_classifier(train_combined_features, train['label'], test_combined_features, test['label'])


print("\n\nSVM -")
print("Using Sentiment features overlap -")
svm_classifier(train_sentiment, train['label'], test_sentiment, test['label'])
print("Using Count features overlap -")
svm_classifier(train_overlap, train['label'], test_overlap, test['label'])
print("Using Count features normalized overlap -")
svm_classifier(train_normalized_overlap, train['label'], test_normalized_overlap, test['label'])
print("Using Tfidf features -")
svm_classifier(train_tfidf_similarity, train['label'], test_tfidf_similarity, test['label'])
print("Combining all features - ")
svm_classifier(train_combined_features, train['label'], test_combined_features, test['label'])


print("\n\nAdaboost -")
print("Using Sentiment features overlap -")
adaboost_classifier(train_sentiment, train['label'], test_sentiment, test['label'])
print("Using Count features overlap -")
adaboost_classifier(train_overlap, train['label'], test_overlap, test['label'])
print("Using Count features normalized overlap -")
adaboost_classifier(train_normalized_overlap, train['label'], test_normalized_overlap, test['label'])
print("Using Tfidf features -")
adaboost_classifier(train_tfidf_similarity, train['label'], test_tfidf_similarity, test['label'])
print("Combining all features - ")
adaboost_classifier(train_combined_features, train['label'], test_combined_features, test['label'])


print("\n\nBagging -")
print("Using Sentiment features overlap -")
bagging_classifier(train_sentiment, train['label'], test_sentiment, test['label'])
print("Using Count features overlap -")
bagging_classifier(train_overlap, train['label'], test_overlap, test['label'])
print("Using Count features normalized overlap -")
bagging_classifier(train_normalized_overlap, train['label'], test_normalized_overlap, test['label'])
print("Using Tfidf features -")
bagging_classifier(train_tfidf_similarity, train['label'], test_tfidf_similarity, test['label'])
print("Combining all features - ")
bagging_classifier(train_combined_features, train['label'], test_combined_features, test['label'])


print("\n\nXGBoost -")
print("Using Sentiment features overlap -")
xgboost_classifier(train_sentiment, train['label'], test_sentiment, test['label'])
print("Using Count features overlap -")
xgboost_classifier(train_overlap, train['label'], test_overlap, test['label'])
print("Using Count features normalized overlap -")
xgboost_classifier(train_normalized_overlap, train['label'], test_normalized_overlap, test['label'])
print("Using Tfidf features -")
xgboost_classifier(train_tfidf_similarity, train['label'], test_tfidf_similarity, test['label'])
print("Combining all features - ")
xgboost_classifier(train_combined_features, train['label'], test_combined_features, test['label'])
