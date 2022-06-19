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
        tfidf_vectorizer = TfidfVectorizer(analyzer=hindi_analyzer)
        tfidf_fit = tfidf_vectorizer.fit(texts)
        return tfidf_fit

    def tfidf_vec(self, fit, texts):
        tfidf_transform = fit.transform(texts)
        return tfidf_transform.toarray()

    def cosine_similarity(self, head, body):
        dot = np.zeros((len(head)))
        for i in range(len(head)):
            if norm(head[i]) == 0 or norm(body[i]) == 0:
                dot[i] = 0
            else:
                dot[i] = np.dot(head[i], body[i]) / \
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
        dot = np.zeros((len(head)))
        for i in range(len(head)):
            if norm(head[i]) == 0 or norm(body[i]) == 0:
                dot[i] = 0
            else:
                dot[i] = np.dot(head[i], body[i]) / \
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

    results_file.write("Accuracy using xgb is" +
                       str(accuracy_score(y_test, best_preds)) + "\n")
    results_file.write("F1-score using xgb is" +
                       str(f1_score(y_test, best_preds, average=None)) + "\n")


def return_features(df):
    combine_head_body(df)
    if use_sentiment_features:
        sentiment_features = np.zeros((len(df), 1))
    overlap = np.zeros((len(df), 1))
    normalized_overlap = np.zeros((len(df), 1))
    tfidf_similarity = np.zeros((len(df), 1))

    for i in tqdm(range(len(df))):
        article = df[i:i+1]

        if use_sentiment_features:
            sentiment = SentimentFeatures()
            bert = sentiment.model(bert_dir)
            sentiment_head, _ = bert.predict(list(article['heading']))
            sentiment_body, _ = bert.predict(list(article['body']))

            sentiment_features[i] = sentiment.similarity(
                sentiment_head, sentiment_body)

        cf = CountFeatures()
        fit = cf.fit(article['combined'])
        count_head = cf.count_vec(fit, article['heading'])
        count_body = cf.count_vec(fit, article['body'])

        overlap[i] = cf.overlap(count_head, count_body)
        normalized_overlap[i] = cf.normalized_overlap(count_head, count_body)

        tf = TfidfFeatures()
        fit = tf.fit(article['combined'])
        tfidf_combined = tf.tfidf_vec(fit, article['combined'])
        tfidf_head = tf.tfidf_vec(fit, article['heading'])
        tfidf_body = tf.tfidf_vec(fit, article['body'])

        tfidf_similarity[i] = tf.cosine_similarity(tfidf_head, tfidf_body)

    if use_sentiment_features:
        combined_features = np.concatenate(
            (sentiment_features, overlap, normalized_overlap, tfidf_similarity), axis=1)
        return sentiment_features, overlap, normalized_overlap, tfidf_similarity, combined_features
    else:
        combined_features = np.concatenate(
            (overlap, normalized_overlap, tfidf_similarity), axis=1)
        return overlap, normalized_overlap, tfidf_similarity, combined_features


def run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features, test_overlap, test_normalized_overlap, test_tfidf_similarity, test_combined_features, str, train_sentiment=None, test_sentiment=None):
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
    results_file.write("Combining all features -\n")
    xgboost_classifier(train_combined_features,
                       train['label'], test_combined_features, test['label'])


# Main code
parser = argparse.ArgumentParser(
    description='News similarity classifier Arguments')
parser.add_argument('--train_path', type=str, required=True)
parser.add_argument('--dev_path', type=str, required=True)
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
if use_sentiment_features:
    bert_dir = os.path.abspath(args['bert_dir'])

train = pd.read_csv(train_path)
dev = pd.read_csv(dev_path)
test = pd.read_csv(test_path)
gold = pd.read_csv(gold_path)

results_file = open(results_path, 'a')

if use_sentiment_features:
    train_sentiment, train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features = return_features(
        train)
    dev_sentiment, dev_overlap, dev_normalized_overlap, dev_tfidf_similarity, dev_combined_features = return_features(
        dev)
    test_sentiment, test_overlap, test_normalized_overlap, test_tfidf_similarity, test_combined_features = return_features(
        test)
    gold_sentiment, gold_overlap, gold_normalized_overlap, gold_tfidf_similarity, gold_combined_features = return_features(
        gold)

    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features, dev_overlap,
                    dev_normalized_overlap, dev_tfidf_similarity, dev_combined_features, "DEV", train_sentiment, dev_sentiment)
    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features, test_overlap,
                    test_normalized_overlap, test_tfidf_similarity, test_combined_features, "TEST", train_sentiment, test_sentiment)
    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features, gold_overlap,
                    gold_normalized_overlap, gold_tfidf_similarity, gold_combined_features, "GOLD", train_sentiment, gold_sentiment)
else:
    train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features = return_features(
        train)
    dev_overlap, dev_normalized_overlap, dev_tfidf_similarity, dev_combined_features = return_features(
        dev)
    test_overlap, test_normalized_overlap, test_tfidf_similarity, test_combined_features = return_features(
        test)
    gold_overlap, gold_normalized_overlap, gold_tfidf_similarity, gold_combined_features = return_features(
        gold)

    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features,
                    dev_overlap, dev_normalized_overlap, dev_tfidf_similarity, dev_combined_features, "DEV")
    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features,
                    test_overlap, test_normalized_overlap, test_tfidf_similarity, test_combined_features, "TEST")
    run_classifiers(train_overlap, train_normalized_overlap, train_tfidf_similarity, train_combined_features,
                    gold_overlap, gold_normalized_overlap, gold_tfidf_similarity, gold_combined_features, "GOLD")
