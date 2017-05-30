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
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(clf, X, y, name):
	y_pred = clf.predict(X)
	print('Confusion matrix for ' + name)
	cm = confusion_matrix(y, y_pred)
	print(cm)
	return cm

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

def neural_network(X, y,layers=(50,50),activation='relu',solver='adam'):
	print("\n*** Initing Neural Network classifier ***")

	# min_max_scaler = MinMaxScaler()
	# X = min_max_scaler.fit_transform(X)

	clf = MLPClassifier(hidden_layer_sizes=layers,activation=activation,solver=solver)
	scores_accuracy = cross_val_score(clf,X,y,scoring='accuracy',cv=3,n_jobs=8)
	scores_logloss = cross_val_score(clf,X,y,scoring='neg_log_loss',cv=3,n_jobs=8)

	print("NN accuracy: %0.4f (+/- %0.4f)" % (scores_accuracy.mean()*100, scores_accuracy.std()*100))
	print("NN log_loss: %0.4f (+/- %0.4f)" % (-scores_logloss.mean(), scores_logloss.std()))

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=1)
	clf.fit(X_train,y_train)
	y_pred = clf.predict_proba(X_test)
	print("NN validation accuracy: ", clf.score(X_test,y_test))
	print("NN validation log_loss: ", log_loss(y_test,y_pred))

	generate_confusion_matrix(clf, X_test, y_test, 'Neural network')

	return clf

def xgboost(X, y):
	print("\n*** Initing XGBooster classifier ***")

	clf = xgb.XGBClassifier(learning_rate=0.15, n_estimators=500, nthread=8, max_depth=6, seed=0, silent=True)
	# scores_accuracy = cross_val_score(clf,X,y,scoring='accuracy',cv=3)
	# scores_logloss = cross_val_score(clf,X,y,scoring='neg_log_loss',cv=3)

	# print("XGBoost accuracy: %0.4f (+/- %0.4f)" % (scores_accuracy.mean()*100, scores_accuracy.std()*100))
	# print("XGBoost log_loss: %0.4f (+/- %0.4f)" % (-scores_logloss.mean(), scores_logloss.std()))

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=1)
	clf.fit(X_train,y_train)
	y_pred = clf.predict_proba(X_test)
	print("XGBoost validation accuracy: ", clf.score(X_test,y_test))
	print("XGBoost validation log_loss: ", log_loss(y_test,y_pred))

	generate_confusion_matrix(clf, X_test, y_test, 'XGBoost')

	return clf

# training with crossvalidation
def naive_bayes_clf_cv(X, y, clf, name):
	scores_accuracy = cross_val_score(clf,X,y,scoring='accuracy',cv=3,n_jobs=6)
	scores_logloss = cross_val_score(clf,X,y,scoring='neg_log_loss',cv=3,n_jobs=6)

	print("Naive Bayes " + name + " accuracy: %0.4f (+/- %0.4f)" % (scores_accuracy.mean()*100, scores_accuracy.std()*100))
	print("Naive Bayes " + name + " log_loss: %0.4f (+/- %0.4f)" % (-scores_logloss.mean(), scores_logloss.std()))

# fit to test validation set
def naive_bayes_clf_val(X_train, X_test, y_train, y_test, clf, name):
	clf.fit(X_train, y_train)
	y_pred = clf.predict_proba(X_test)
	print('Naive bayes ' + name + ' validation score: ', (clf.score(X_test, y_test))*100)
	print('Naive bayes ' + name + ' validation log loss: ', log_loss(y_test, y_pred))

	generate_confusion_matrix(clf,X_test,y_test, 'Naive bayes ' + name)

	return clf

def naive_bayes(X, y):
	print("\n*** Initing naive bayes classifiers ***")

	min_max_scaler = MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	clf1 = GaussianNB()
	clf2 = MultinomialNB()
	clf3 = BernoulliNB()
	naive_bayes_clf_cv(X, y, clf1, 'gaussian')
	naive_bayes_clf_cv(X, y, clf2, 'multinomial')
	naive_bayes_clf_cv(X, y, clf3, 'bernoulli')

	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=1)
	clf1 = naive_bayes_clf_val(X_train, X_test, y_train, y_test, clf1, 'gaussian')
	clf2 = naive_bayes_clf_val(X_train, X_test, y_train, y_test, clf2, 'multinomial')
	clf3 = naive_bayes_clf_val(X_train, X_test, y_train, y_test, clf3, 'bernoulli')

	return {'naive_bayes_gaussian': clf1, 'naive_bayes_multinomial': clf2, 'naive_bayes_bernoulli': clf3}

def classify(nn_learning=False,xgb_learning=False,nb_learning=False,submission=False):
	filename = Constants.TRAIN_MATRIX_FEATURES
	# filename = Constants.TRAIN_MATRIX_FEATURES+'_semantic.csv'
	# filename = Constants.TRAIN_MATRIX_FEATURES+'_semantic_svd.csv'
	print(filename)

	X, y = load_train_features(filename)
	clf_dict = {}

	n=50000

	if(n>0):
		X = X[:n]
		y = y[:n]

	if(nb_learning):
		clf_nb_dict = naive_bayes(X, y)
		clf_dict.update(clf_nb_dict)
	if(xgb_learning):
		clf_xgb = xgboost(X, y)
		clf_dict['xgboost'] = clf_xgb
	if(nn_learning):
		clf_nn = neural_network(X, y)
		clf_dict['neural_network'] = clf_nn

	if(submission):
		X_Kaggle = load_test_features(Constants.TEST_MATRIX_FEATURES)
		for clf_name in clf_dict:
			clf = clf_dict[clf_name]
			Util.generate_submission(X_Kaggle, clf, clf_name+'.csv')