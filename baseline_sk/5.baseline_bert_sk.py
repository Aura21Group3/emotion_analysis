from sklearn.externals import joblib
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import time
import numpy as np
from sklearn.model_selection import train_test_split

start = time.clock()
X_train_tfidf_vec= joblib.load('./data/X_train_bert_vec')
X_test_tfidf_vec = joblib.load('./data/X_test_bert_vec')
y_train = joblib.load('./data/y_train_one_hot')
y_test = joblib.load('./data/y_test_one_hot')
print(X_train_tfidf_vec.shape)
print(X_test_tfidf_vec.shape)
print(y_train.shape)
print(y_test.shape)
# clf = GradientBoostingClassifier(n_estimators=200)
# clf = RandomForestClassifier(n_estimators=50)
clf = LinearSVC()
# clf = LogisticRegression(penalty='l2')
y_train = np.array([ np.argmax(i) for i in y_train])
y_test = np.array([ np.argmax(i) for i in y_test])

# clf = GaussianNB()
X_train,X_test,y_train,y_test = train_test_split(X_train_tfidf_vec,y_train,test_size=0.2)
clf.fit(X_train,y_train)
print('训练集得分:{}'.format(clf.score(X_train,y_train)))
print('测试集得分:{}'.format(clf.score(X_test,y_test)))
end = time.clock()
print('训练用时:{}'.format(end - start))



