import logger

import pandas as pd
import numpy as np
from numpy import ravel

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class_weight = 'balanced'


def read_data(path):
    df = pd.read_csv(path).fillna(0)
    if df.shape[0] == 0:
        return None
    logger.log(df.shape)
    logger.log(df)
    target = ravel(df[[run_type + "_level"]].astype('int').to_numpy())
    data = df.drop([run_type + "_level", "uid"], axis=1).to_dict("records")
    return data, target


def to_feature(data, target):
    fe = DictVectorizer()
    feature = fe.fit_transform(data).toarray()
    logger.log(feature)
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(target)
    logger.log(label)
    return feature, label


def to_npy():
    data, target = read_data("data/processed_data/" + run_type + "_train.csv")
    predict_data, no_need = read_data("data/processed_data/" + run_type + "_test.csv")
    print(target)
    feature_data, label_data = to_feature(data + predict_data, target)
    train_feature = feature_data[:len(data)]
    predict_feature = feature_data[-len(predict_data):]
    print(len(data))
    print(len(predict_data))
    print(len(train_feature))
    print(len(predict_feature))
    np.save("data/npy/" + run_type + "_train_feature", train_feature)
    np.save("data/npy/" + run_type + "_train_label", label_data)
    np.save("data/npy/" + run_type + "_predict_feature", predict_feature)


def load_npy(feature_path, label_path):
    train_feature = np.load(feature_path)
    print(train_feature)
    train_label = np.load(label_path)
    print(train_label)
    return train_feature, train_label


def divide_train_test(feature, label):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.25)
    logger.log("Length of x_train: " + str(len(x_train)))
    logger.log("Length of y_train: " + str(len(y_train)))
    logger.log("Length of x_test:  " + str(len(x_test)))
    logger.log("Length of y_test:  " + str(len(y_test)))
    return x_train, x_test, y_train, y_test


def voting_train(x_train, x_test, y_train, y_test):
    clf1 = LogisticRegression(max_iter=10000)
    clf2 = GaussianNB()
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

    vote_clf = VotingClassifier(estimators=[('lr', clf1), ('GNB', clf2), ('SVM', clf3), ('RFC', clf4)], voting='soft')
    # vote_clf = VotingClassifier(estimators=[('SVM', clf3), ('RFC', clf4)], voting='soft')

    vote_clf.fit(x_train, y_train)
    joblib.dump(vote_clf, "data/model/" + run_type + "_2" + "vote.model")
    y_predict = vote_clf.predict(x_test)
    logger.log(y_predict)
    score = vote_clf.score(x_test, y_test)
    logger.log(score)
    confusion = confusion_matrix(y_test, y_predict)
    logger.log(confusion)
    report = classification_report(y_test, y_predict)
    logger.log(report)


def single_train(clf_type, x_train, x_test, y_train, y_test):
    if clf_type == "LR":
        clf = LogisticRegression(max_iter=10000, class_weight=class_weight)
    elif clf_type == "GNB":
        clf = GaussianNB()
    elif clf_type == "SVM":
        clf = SVC(kernel='rbf', probability=True, class_weight=class_weight)
    elif clf_type == "RFC":
        clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, class_weight=class_weight)
    else:
        clf = LogisticRegression(max_iter=10000, class_weight=class_weight)
    clf.fit(x_train, y_train)
    joblib.dump(clf, "data/model/" + run_type + "_" + clf_type + ".model")
    y_predict = clf.predict(x_test)
    logger.log(y_predict)
    score = clf.score(x_test, y_test)
    logger.log("score: " + str(score))
    confusion = confusion_matrix(y_test, y_predict)
    logger.log(confusion)
    report = classification_report(y_test, y_predict)
    logger.log(report)


run_type = "star"

if __name__ == '__main__':
    # fake_data = load_iris()
    # feature_data, label_data = fake_data.data, fake_data.target

    # to_npy()
    feature_data, label_data = load_npy("data/npy/" + run_type + "_train_feature.npy",
                                        "data/npy/" + run_type + "_train_label.npy")
    x_train, x_test, y_train, y_test = divide_train_test(feature_data, label_data)

    logger.log("=" * 50 + "vote" + "=" * 50)
    voting_train(x_train, x_test, y_train, y_test)
    # logger.log("=" * 50 + "LR" + "=" * 50)
    # single_train("LR", x_train, x_test, y_train, y_test)
    # logger.log("=" * 50 + "GNB" + "=" * 50)
    # single_train("GNB", x_train, x_test, y_train, y_test)
    # logger.log("=" * 50 + "SVM" + "=" * 50)
    # single_train("SVM", x_train, x_test, y_train, y_test)
    # logger.log("=" * 50 + "RFC" + "=" * 50)
    # single_train("RFC", x_train, x_test, y_train, y_test)
