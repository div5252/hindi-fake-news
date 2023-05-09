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
        ans = ans.reshape(-1, 1)
        return ans


class CountFeatures():
    def fit(self, texts):
        count_vectorizer = CountVectorizer(analyzer=hindi_analyzer)
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
        ans = np.zeros((len(head), 1))
        for i in range(len(bool_head)):
            ans[i][0] = np.dot(bool_head[i], bool_body[i])
        return ans

    def normalized_overlap(self, head, body):
        ans = np.zeros((len(head), 1))
        for i in range(len(head)):
            for j in range(len(head[i])):
                if head[i][j] == 0:
                    ans[i][0] += 0
                else:
                    ans[i][0] += float(body[i][j]) / float(head[i][j])
        return ans


class TfidfFeatures():
    def fit(self, texts):
        tfidf_vectorizer = TfidfVectorizer(analyzer=hindi_analyzer)
        tfidf_fit = tfidf_vectorizer.fit(texts)
        return tfidf_fit

    def tfidf_vec(self, fit, texts):
        tfidf_transform = fit.transform(texts)
        return tfidf_transform.toarray()

    def cosine_similarity(self, head, body):
        dot = np.zeros((len(head), 1))
        for i in range(len(head)):
            if norm(head[i]) == 0 or norm(body[i]) == 0:
                dot[i][0] = 0
            else:
                dot[i][0] = np.dot(head[i], body[i]) / \
                    (norm(head[i]) * norm(body[i]))
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
        dot = np.zeros((len(head), 1))
        for i in range(len(head)):
            if norm(head[i]) == 0 or norm(body[i]) == 0:
                dot[i][0] = 0
            else:
                dot[i][0] = np.dot(head[i], body[i]) / \
                    (norm(head[i]) * norm(body[i]))
        return dot


# Classification using Decision Tree
def dt_classifier(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_predicted = dt.predict(X_test)
    results_file.write("Accuracy using Decision Tree is " +
                       str(accuracy_score(y_test, dt_predicted)) + "\n")
    results_file.write("F1-score using Decision Tree is " +
                       str(f1_score(y_test, dt_predicted, average=None)) + "\n")


# Classification using SVM
def svm_classifier(X_train, y_train, X_test, y_test):
    svm = LinearSVC(max_iter=10000)
    svm.fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    results_file.write("Accuracy using SVM is " +
                       str(accuracy_score(y_test, svm_predicted)) + "\n")
    results_file.write("F1-score using SVM is " +
                       str(f1_score(y_test, svm_predicted, average=None)) + "\n")


# Classification using AdaBoost
def adaboost_classifier(X_train, y_train, X_test, y_test):
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    ada_predicted = ada.predict(X_test)
    results_file.write("Accuracy using AdaBoost is " +
                       str(accuracy_score(y_test, ada_predicted)) + "\n")
    results_file.write("F1-score using AdaBoost is " +
                       str(f1_score(y_test, ada_predicted, average=None)) + "\n")

# Classification using bagging
def bagging_classifier(X_train, y_train, X_test, y_test):
    bag = BaggingClassifier()
    bag.fit(X_train, y_train)
    bag_predicted = bag.predict(X_test)
    results_file.write("Accuracy using Bagging is " +
                       str(accuracy_score(y_test, bag_predicted)) + "\n")
    results_file.write("F1-score using Bagging is " +
                       str(f1_score(y_test, bag_predicted, average=None)) + "\n")

# Classification using XGBoost
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

    results_file.write("Accuracy using xgb is " +
                       str(accuracy_score(y_test, best_preds)) + "\n")
    results_file.write("F1-score using xgb is " +
                       str(f1_score(y_test, best_preds, average=None)) + "\n")


def run_classifiers(train, train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features, test, test_overlap, test_normalized_overlap, test_tfidf_similarity, test_svd_similarity, test_combined_features, str, train_sentiment=None, test_sentiment=None):
    results_file.write("\n\nON " + str + " SET:\n\n")

    results_file.write("Decision Tree Classifier\n")
    if use_sentiment_features:
        results_file.write("Using Sentiment features overlap -\n")
        dt_classifier(train_sentiment,
                      train['label'], test_sentiment, test['label'])
    results_file.write("Using Count features overlap -\n")
    dt_classifier(train_overlap, train['label'], test_overlap, test['label'])
    results_file.write("Using Count features normalized overlap -\n")
    dt_classifier(train_normalized_overlap,
                  train['label'], test_normalized_overlap, test['label'])
    results_file.write("Using Tfidf features -\n")
    dt_classifier(train_tfidf_similarity,
                  train['label'], test_tfidf_similarity, test['label'])
    results_file.write("Using SVD features -\n")
    dt_classifier(train_svd_similarity,
                  train['label'], test_svd_similarity, test['label'])
    results_file.write("Combining all features -\n")
    dt_classifier(train_combined_features,
                  train['label'], test_combined_features, test['label'])

    results_file.write("\n\nSVM Classifier\n")
    if use_sentiment_features:
        results_file.write("Using Sentiment features overlap -\n")
        svm_classifier(train_sentiment,
                       train['label'], test_sentiment, test['label'])
    results_file.write("Using Count features overlap -\n")
    svm_classifier(train_overlap, train['label'], test_overlap, test['label'])
    results_file.write("Using Count features normalized overlap -\n")
    svm_classifier(train_normalized_overlap,
                   train['label'], test_normalized_overlap, test['label'])
    results_file.write("Using Tfidf features -\n")
    svm_classifier(train_tfidf_similarity,
                   train['label'], test_tfidf_similarity, test['label'])
    results_file.write("Using SVD features -\n")
    svm_classifier(train_svd_similarity,
                   train['label'], test_svd_similarity, test['label'])
    results_file.write("Combining all features -\n")
    svm_classifier(train_combined_features,
                   train['label'], test_combined_features, test['label'])

    results_file.write("\n\nAdaboost Classifier\n")
    if use_sentiment_features:
        results_file.write("Using Sentiment features overlap -\n")
        adaboost_classifier(
            train_sentiment, train['label'], test_sentiment, test['label'])
    results_file.write("Using Count features overlap -\n")
    adaboost_classifier(
        train_overlap, train['label'], test_overlap, test['label'])
    results_file.write("Using Count features normalized overlap -\n")
    adaboost_classifier(train_normalized_overlap,
                        train['label'], test_normalized_overlap, test['label'])
    results_file.write("Using Tfidf features -\n")
    adaboost_classifier(train_tfidf_similarity,
                        train['label'], test_tfidf_similarity, test['label'])
    results_file.write("Using SVD features -\n")
    adaboost_classifier(train_svd_similarity,
                        train['label'], test_svd_similarity, test['label'])
    results_file.write("Combining all features -\n")
    adaboost_classifier(train_combined_features,
                        train['label'], test_combined_features, test['label'])

    results_file.write("\n\nBagging Classifier\n")
    if use_sentiment_features:
        results_file.write("Using Sentiment features overlap -\n")
        bagging_classifier(
            train_sentiment, train['label'], test_sentiment, test['label'])
    results_file.write("Using Count features overlap -\n")
    bagging_classifier(
        train_overlap, train['label'], test_overlap, test['label'])
    results_file.write("Using Count features normalized overlap -\n")
    bagging_classifier(train_normalized_overlap,
                       train['label'], test_normalized_overlap, test['label'])
    results_file.write("Using Tfidf features -\n")
    bagging_classifier(train_tfidf_similarity,
                       train['label'], test_tfidf_similarity, test['label'])
    results_file.write("Using SVD features -\n")
    bagging_classifier(train_svd_similarity,
                       train['label'], test_svd_similarity, test['label'])
    results_file.write("Combining all features -\n")
    bagging_classifier(train_combined_features,
                       train['label'], test_combined_features, test['label'])

    results_file.write("\n\nXGBoost Classifier\n")
    if use_sentiment_features:
        results_file.write("Using Sentiment features overlap -\n")
        xgboost_classifier(
            train_sentiment, train['label'], test_sentiment, test['label'])
    results_file.write("Using Count features overlap -\n")
    xgboost_classifier(
        train_overlap, train['label'], test_overlap, test['label'])
    results_file.write("Using Count features normalized overlap -\n")
    xgboost_classifier(train_normalized_overlap,
                       train['label'], test_normalized_overlap, test['label'])
    results_file.write("Using Tfidf features -\n")
    xgboost_classifier(train_tfidf_similarity,
                       train['label'], test_tfidf_similarity, test['label'])
    results_file.write("Using SVD features -\n")
    xgboost_classifier(train_svd_similarity,
                       train['label'], test_svd_similarity, test['label'])
    results_file.write("Combining all features -\n")
    xgboost_classifier(train_combined_features,
                       train['label'], test_combined_features, test['label'])


# Main code
parser = argparse.ArgumentParser(
    description='News similarity classifier Arguments')
parser.add_argument('--train_path', type=str, required=True)
parser.add_argument('--use_dev', type=int, default=0)
parser.add_argument('--dev_path', type=str, required=False, default='')
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--gold_path', type=str, required=True)
parser.add_argument('--results_path', type=str, required=True)
parser.add_argument('--use_sentiment_features',
                    type=int, default=0, required=False)
parser.add_argument('--bert_dir', type=str, required=False)

args = vars(parser.parse_args())

train_path = os.path.abspath(args['train_path'])
dev_path = os.path.abspath(args['dev_path'])
test_path = os.path.abspath(args['test_path'])
gold_path = os.path.abspath(args['gold_path'])
results_path = os.path.abspath(args['results_path'])
use_sentiment_features = int(args['use_sentiment_features'])
use_dev = int(args['use_dev'])
if use_sentiment_features:
    bert_dir = os.path.abspath(args['bert_dir'])

train = pd.read_csv(train_path)
if use_dev == 1:
    dev = pd.read_csv(dev_path)
test = pd.read_csv(test_path)
gold = pd.read_csv(gold_path)

if use_dev == 1:
    combined = pd.concat([train, dev, test, gold], ignore_index=True)
else:
    combined = pd.concat([train, test, gold], ignore_index=True)
combine_head_body(train)
if use_dev == 1:
    combine_head_body(dev)
combine_head_body(test)
combine_head_body(gold)
combine_head_body(combined)

if use_sentiment_features:
    sentiment = SentimentFeatures()
    bert = sentiment.model(bert_dir)
    sentiment_train_head, _ = bert.predict(list(train['heading']))
    sentiment_train_body, _ = bert.predict(list(train['body']))
    if use_dev == 1:
        sentiment_dev_head, _ = bert.predict(list(dev['heading']))
        sentiment_dev_body, _ = bert.predict(list(dev['body']))
    sentiment_test_head, _ = bert.predict(list(test['heading']))
    sentiment_test_body, _ = bert.predict(list(test['body']))
    sentiment_gold_head, _ = bert.predict(list(gold['heading']))
    sentiment_gold_body, _ = bert.predict(list(gold['body']))

    train_sentiment = sentiment.similarity(
        sentiment_train_head, sentiment_train_body)
    if use_dev == 1:
        dev_sentiment = sentiment.similarity(
        sentiment_dev_head, sentiment_dev_body)
    test_sentiment = sentiment.similarity(
        sentiment_test_head, sentiment_test_body)
    gold_sentiment = sentiment.similarity(
        sentiment_gold_head, sentiment_gold_body)

cf = CountFeatures()
fit = cf.fit(combined['combined'])
count_train_head = cf.count_vec(fit, train['heading'])
count_train_body = cf.count_vec(fit, train['body'])
if use_dev == 1:
    count_dev_head = cf.count_vec(fit, dev['heading'])
    count_dev_body = cf.count_vec(fit, dev['body'])
count_test_head = cf.count_vec(fit, test['heading'])
count_test_body = cf.count_vec(fit, test['body'])
count_gold_head = cf.count_vec(fit, gold['heading'])
count_gold_body = cf.count_vec(fit, gold['body'])

train_overlap = cf.overlap(count_train_head, count_train_body)
train_normalized_overlap = cf.normalized_overlap(
    count_train_head, count_train_body)
if use_dev == 1:
    dev_overlap = cf.overlap(count_dev_head, count_dev_body)
    dev_normalized_overlap = cf.normalized_overlap(count_dev_head, count_dev_body)
test_overlap = cf.overlap(count_test_head, count_test_body)
test_normalized_overlap = cf.normalized_overlap(
    count_test_head, count_test_body)
gold_overlap = cf.overlap(count_gold_head, count_gold_body)
gold_normalized_overlap = cf.normalized_overlap(
    count_gold_head, count_gold_body)

tf = TfidfFeatures()
fit = tf.fit(combined['combined'])
tfidf_combined = tf.tfidf_vec(fit, combined['combined'])
tfidf_train_head = tf.tfidf_vec(fit, train['heading'])
tfidf_train_body = tf.tfidf_vec(fit, train['body'])
if use_dev == 1:
    tfidf_dev_head = tf.tfidf_vec(fit, dev['heading'])
    tfidf_dev_body = tf.tfidf_vec(fit, dev['body'])
tfidf_test_head = tf.tfidf_vec(fit, test['heading'])
tfidf_test_body = tf.tfidf_vec(fit, test['body'])
tfidf_gold_head = tf.tfidf_vec(fit, gold['heading'])
tfidf_gold_body = tf.tfidf_vec(fit, gold['body'])

train_tfidf_similarity = tf.cosine_similarity(
    tfidf_train_head, tfidf_train_body)
if use_dev == 1:
    dev_tfidf_similarity = tf.cosine_similarity(tfidf_dev_head, tfidf_dev_body)
test_tfidf_similarity = tf.cosine_similarity(tfidf_test_head, tfidf_test_body)
gold_tfidf_similarity = tf.cosine_similarity(tfidf_gold_head, tfidf_gold_body)

sf = SvdFeatures()
fit = sf.fit(tfidf_combined)
svd_train_head = sf.svd_vec(fit, tfidf_train_head)
svd_train_body = sf.svd_vec(fit, tfidf_train_body)
if use_dev == 1:
    svd_dev_head = sf.svd_vec(fit, tfidf_dev_head)
    svd_dev_body = sf.svd_vec(fit, tfidf_dev_body)
svd_test_head = sf.svd_vec(fit, tfidf_test_head)
svd_test_body = sf.svd_vec(fit, tfidf_test_body)
svd_gold_head = sf.svd_vec(fit, tfidf_gold_head)
svd_gold_body = sf.svd_vec(fit, tfidf_gold_body)

train_svd_similarity = np.asarray(
    sf.cosine_similarity(svd_train_head, svd_train_body))
if use_dev == 1:
    dev_svd_similarity = np.asarray(
    sf.cosine_similarity(svd_dev_head, svd_dev_body))
test_svd_similarity = np.asarray(
    sf.cosine_similarity(svd_test_head, svd_test_body))
gold_svd_similarity = np.asarray(
    sf.cosine_similarity(svd_gold_head, svd_gold_body))

if use_sentiment_features:
    train_combined_features = np.concatenate(
        (train_sentiment, train_overlap, train_normalized_overlap, train_tfidf_similarity), axis=1)
    if use_dev == 1:
        dev_combined_features = np.concatenate(
        (dev_sentiment, dev_overlap, dev_normalized_overlap, dev_tfidf_similarity), axis=1)
    test_combined_features = np.concatenate(
        (test_sentiment, test_overlap, test_normalized_overlap, test_tfidf_similarity), axis=1)
    gold_combined_features = np.concatenate(
        (gold_sentiment, gold_overlap, gold_normalized_overlap, gold_tfidf_similarity), axis=1)
else:
    train_combined_features = np.concatenate(
        (train_overlap, train_normalized_overlap, train_tfidf_similarity), axis=1)
    if use_dev == 1:
        dev_combined_features = np.concatenate(
        (dev_overlap, dev_normalized_overlap, dev_tfidf_similarity), axis=1)
    test_combined_features = np.concatenate(
        (test_overlap, test_normalized_overlap, test_tfidf_similarity), axis=1)
    gold_combined_features = np.concatenate(
        (gold_overlap, gold_normalized_overlap, gold_tfidf_similarity), axis=1)

results_file = open(results_path, 'a')

if use_sentiment_features:
    if use_dev == 1:
        run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features, dev_overlap,
                    dev_normalized_overlap, dev_tfidf_similarity, dev_svd_similarity, dev_combined_features, "DEV", train_sentiment, dev_sentiment)
    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features, test_overlap,
                    test_normalized_overlap, test_tfidf_similarity, test_svd_similarity, test_combined_features, "TEST", train_sentiment, test_sentiment)
    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features, gold_overlap,
                    gold_normalized_overlap, gold_tfidf_similarity, gold_svd_similarity, gold_combined_features, "GOLD", train_sentiment, gold_sentiment)
else:
    if use_dev == 1:
        run_classifiers(train, train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features,
                    dev, dev_overlap, dev_normalized_overlap, dev_tfidf_similarity, dev_svd_similarity, dev_combined_features, "DEV")
    run_classifiers(train, train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features,
                    test, test_overlap, test_normalized_overlap, test_tfidf_similarity, test_svd_similarity, test_combined_features, "TEST")
    run_classifiers(train, train_overlap, train_normalized_overlap, train_tfidf_similarity, train_svd_similarity, train_combined_features,
                    gold, gold_overlap, gold_normalized_overlap, gold_tfidf_similarity, gold_svd_similarity, gold_combined_features, "GOLD")
