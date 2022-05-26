from sklearn.ensemble import VotingClassifier  # 导入投票分类器
from sklearn.datasets import load_iris  # 导入训练数据集

# 构建若干基础模型
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = load_iris()
print(iris)
clf1 = LogisticRegression()
clf2 = GaussianNB()
clf3 = SVC(kernel='rbf')

# 实例化
vote_clf = VotingClassifier(estimators=[('lr', clf1), ('GNB', clf2), ('SVM', clf3)], voting='soft')

# 训练投票分类器
vote_clf.fit(iris.data, iris.target)

# # 预测
# vote_clf.predict(iris.data)
#
# # 建立投票分类器(soft)
# clf1 = LogisticRegression()
# clf2 = GaussianNB()
# clf3 = SVC(kernel='rbf', probability=True)
#
# vote_clf2 = VotingClassifier(estimators=[('lr', clf1), ('GNB', clf2), ('SVM', clf3)], voting='soft')  # 实例化
# vote_clf2 = vote_clf2.fit(iris.data, iris.target)  # 训练
# vote_clf2.predict(iris.data)  # 预测

