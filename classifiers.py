import util as Util
from custom_classifiers import *
import csv
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

TRAIN_FEATURES='files/train_features.csv'
TEST_FEATURES='files/test_features.csv'

def load_train_features():
	print('\n*** Loading train features ***')
	X = []
	y = []
	with open(TRAIN_FEATURES, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
		for index, line in enumerate(csv_reader):
			X.append(line[1:len(line)-1])
			y.append(int(line[len(line)-1]))
	print('N_train=',len(X))
	return (X, y)

# Stub X and y
def load_test_features():
	print('\n*** Loading test features ***')
	X = []
	with open(TEST_FEATURES, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
		for index, line in enumerate(csv_reader):
			X.append(line[1:])
	print('N_test=',len(X))
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
	clf = MLPClassifier(hidden_layer_sizes=(100,50))
	clf.fit(X, y)
	y_pred = clf.predict_proba(X)

	print('Neural network log loss: ', log_loss(y, y_pred))
	print('Neural network score: ', (clf.score(X, y))*100)

	return clf

def xgboots(X, y):
	return 0

def classify(random_learning=False,zero_learning=False,one_learning=False,knn_learning=False,neural_learning=False,xgb_learning=False):
	X, y = load_train_features()
	X_Kaggle = load_test_features()

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