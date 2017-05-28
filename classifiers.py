import numpy as np
import util as Util
from custom_classifiers import *
import csv
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn import cross_validation
from random import shuffle
from sklearn.preprocessing import MinMaxScaler
import constants as Constants
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

def load_train_features(features_file):
	print('\n*** Loading train features ***')
	X = []
	y = []
	with open(features_file, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
		for index, line in enumerate(csv_reader):
			X.append(line[1:len(line)-1])
			y.append(int(line[len(line)-1]))
	print('N_train=',len(X))

	c = list(zip(X,y))
	shuffle(c)
	X,y = zip(*c)
	X = np.array(X)
	y = np.array(y)
	
	return (X, y)

# Stub X and y
def load_test_features(features_file):
	print('\n*** Loading test features ***')
	X = []
	with open(features_file, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
		for index, line in enumerate(csv_reader):
			X.append(line[1:])
	print('N_test=',len(X))
	X = np.array(X)
	return X

def dummyOne(X, y):
	print('\n*** Initing "one" classifier ***')
	clf = SimpleClassClassifier(value='1')
	clf.fit(X, y)
	y_pred = clf.predict_proba(X)
	#score = log_loss(y, y_pred)
	#print('Only "1" log loss: ', score)
	return clf

def dummyZero(X, y):
	print('\n*** Initing "zero" classifier ***')
	clf = SimpleClassClassifier(value='0')
	clf.fit(X, y)
	y_pred = clf.predict_proba(X)
	#score = log_loss(y, y_pred)
	#print('Only "0" log loss: ', score)
	return clf

def random(X, y):
	print("\n*** Initing random classifier ***")
	clf = RandomClassifier()
	y_pred = clf.predict_proba(X)
	score_log = log_loss(y, y_pred)
	print('Random log loss: ', score_log)
	return clf

def knn(X, y):
	print("\n*** Initing KNN classifier ***")
	clf = KNeighborsClassifier(n_jobs=4)
	clf.fit(X, y)
	y_pred = clf.predict_proba(X)

	print('KNN log loss: ', log_loss(y, y_pred))
	print('KNN score: ', (1-clf.score(X, y))*100)

	return clf

def neural_network(X, y):
	print("\n*** Initing Neural Network classifier ***")
	clf = MLPClassifier(hidden_layer_sizes=(50))
	clf.fit(X, y)
	y_pred = clf.predict_proba(X)

	print('Neural network log loss: ', log_loss(y, y_pred))
	print('Neural network score: ', (clf.score(X, y))*100)

	return clf

def xgboost(X, y):
	print("\n*** Initing XGBooster classifier ***")
	clf = xgb.XGBClassifier(learning_rate=0.15, n_estimators=200, nthread=8, max_depth=12, seed=0, silent=True)

	scores_accuracy = cross_val_score(clf,X,y,scoring='accuracy',cv=3)
	scores_logloss = cross_val_score(clf,X,y,scoring='log_loss',cv=3)

	print("XGBoost accuracy: %0.4f (+/- %0.4f)" % (scores_accuracy.mean()*100, scores_accuracy.std()))
	print("XGBoost log_loss: %0.4f (+/- %0.4f)" % (scores_logloss.mean()*100, scores_logloss.std()))

	return clf

def SVM(X, y):
	print("\n*** Initing SVM classifier ***")
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=1)
	clf = SVC(probability=True)

	print("Training...")
	clf.fit(X_train, y_train)
	print("Predicting...")
	y_pred = clf.predict_proba(X_train)

	print('SVM log loss (train): ', log_loss(y_train, y_pred))
	print('SVM score (train): ', (clf.score(X_train, y_train))*100)

	y_pred = clf.predict_proba(X_test)
	print('SVM log loss (validation): ', log_loss(y_test, y_pred))
	print('SVM score (validation): ', (clf.score(X_test, y_test))*100)	

	return clf

def naive_bayes(X, y):
	print("\n*** Initing naive bayes classifiers ***")

	min_max_scaler = MinMaxScaler()
	X = min_max_scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=1)
	clf1 = GaussianNB()
	clf2 = MultinomialNB()
	clf3 = BernoulliNB()

	print("Training...")
	clf1.fit(X_train, y_train)
	clf2.fit(X_train, y_train)
	clf3.fit(X_train, y_train)
	print("Predicting...")
	y_pred1 = clf1.predict_proba(X_train)
	y_pred2 = clf2.predict_proba(X_train)
	y_pred3 = clf3.predict_proba(X_train)

	print('Naive bayes gaussiano log loss (train): ', log_loss(y_train, y_pred1))
	print('Naive bayes gaussiano score (train): ', (clf1.score(X_train, y_train))*100)
	print('Naive bayes multinomial log loss (train): ', log_loss(y_train, y_pred2))
	print('Naive bayes multinomial score (train): ', (clf2.score(X_train, y_train))*100)
	print('Naive bayes bernoulli log loss (train): ', log_loss(y_train, y_pred3))
	print('Naive bayes bernoulli score (train): ', (clf3.score(X_train, y_train))*100)

	y_pred1 = clf1.predict_proba(X_test)
	y_pred2 = clf2.predict_proba(X_test)
	y_pred3 = clf3.predict_proba(X_test)
	print('Naive bayes gaussiano log loss (validation): ', log_loss(y_test, y_pred1))
	print('Naive bayes gaussiano score (validation): ', (clf1.score(X_test, y_test))*100)
	print('Naive bayes multinomial log loss (validation): ', log_loss(y_test, y_pred2))
	print('Naive bayes multinomial score (validation): ', (clf2.score(X_test, y_test))*100)
	print('Naive bayes bernoulli log loss (validation): ', log_loss(y_test, y_pred3))
	print('Naive bayes bernoulli score (validation): ', (clf3.score(X_test, y_test))*100)

	return clf1



def classify(random_learning=False,zero_learning=False,one_learning=False,knn_learning=False,neural_learning=False,xgb_learning=False,svm_learning=False,nb_learning=False):
	X, y = load_train_features(Constants.TRAIN_MATRIX_FEATURES)
	# X_Kaggle = load_test_features(Constants.TEST_MATRIX_FEATURES)

	if(random_learning):
		clf_random = random(X, y)
		Util.generate_submission(X_Kaggle, clf_random, 'random_prediction.csv')
	if(zero_learning):
		clf_zero = dummyZero(X, y)
		Util.generate_submission(X_Kaggle, clf_zero, 'all_zero_prediction.csv')
	if(one_learning):
		clf_one = dummyOne(X, y)
		Util.generate_submission(X_Kaggle, clf_one, 'all_one_prediction.csv')
	if(knn_learning):
		clf_knn = knn(X, y)
		Util.generate_submission(X_Kaggle, clf_knn, 'knn_prediction.csv')
	if(neural_learning):
		clf_nn = neural_network(X, y)
		Util.generate_submission(X_Kaggle, clf_nn, 'neural_network_prediction.csv')
	if(xgb_learning):
		clf_xgb = xgboost(X, y)
		# Util.generate_submission(X_Kaggle, clf_xgb, 'xgboost_prediction.csv')
	if(svm_learning):
		clf_svm = SVM(X, y)
		# Util.generate_submission(X_Kaggle, clf_svm, 'svm_prediction.csv')
	if(nb_learning):
		clf_nb = naive_bayes(X, y)
		# Util.generate_submission(X_Kaggle, clf_nb, 'nb_prediction.csv')