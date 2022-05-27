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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def read_data(path):
    df = pd.read_csv(path).fillna(0)
    if df.shape[0] == 0:
        return None
    logger.log(df.shape)
    logger.log(df)
    target = ravel(df[["star_level"]].astype('int').to_numpy())
    data = df.drop(["star_level", "uid"], axis=1).to_dict("records")
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
    data, target = read_data("data/processed_data/star_train.csv")
    predict_data, no_need = read_data("data/processed_data/star_test.csv")
    print(target)
    feature_data, label_data = to_feature(data + predict_data, target)
    train_feature = feature_data[:len(data)]
    predict_feature = feature_data[-len(predict_data):]
    print(len(data))
    print(len(predict_data))
    print(len(train_feature))
    print(len(predict_feature))
    np.save("data/npy/star_train_feature", train_feature)
    np.save("data/npy/star_train_label", label_data)
    np.save("data/npy/star_predict_feature", predict_feature)


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
    # scalier = preprocessing.StandardScaler().fit(x_train)
    # data_scaled = scalier.transform(x_train)
    clf1 = LogisticRegression(max_iter=50000)
    clf2 = GaussianNB()
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

    vote_clf = VotingClassifier(estimators=[('lr', clf1), ('GNB', clf2), ('SVM', clf3), ('RFC', clf4)], voting='soft')

    vote_clf.fit(x_train, y_train)
    joblib.dump(vote_clf, "data/model/" + run_type + "_" + "vote.model")
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
        clf = LogisticRegression(max_iter=50000)
    elif clf_type == "GNB":
        clf = GaussianNB()
    elif clf_type == "SVM":
        clf = SVC(kernel='rbf', probability=True)
    elif clf_type == "RFC":
        clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=10000)
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


def print_np(array):
    i = 0
    for x in array:
        i += 1
        if i == 500:
            break
        print(x, end=", ")
    print()


run_type = "star"

if __name__ == '__main__':
    # to_npy()
    feature_data, label_data = load_npy("data/npy/star_train_feature.npy", "data/npy/star_train_label.npy")
    # fake_data = load_iris()
    # feature_data, label_data = fake_data.data, fake_data.target
    x_train, x_test, y_train, y_test = divide_train_test(feature_data, label_data)

    voting_train(x_train, x_test, y_train, y_test)
    single_train("LR", x_train, x_test, y_train, y_test)
    single_train("GNB", x_train, x_test, y_train, y_test)
    single_train("SVM", x_train, x_test, y_train, y_test)
    single_train("RFC", x_train, x_test, y_train, y_test)
    # print(label_data)
    # predict("data/model/vote.model", feature_data)
