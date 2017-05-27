from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import csv
import nltk
import distance
import util as Util

TRAIN_FILE='files/train.csv'
TEST_FILE='files/test.csv'
TRAIN_FEATURES='files/train_features.csv'
TEST_FEATURES='files/test_features.csv'

def tokenize_question(q):
	sentences = sent_tokenize(q)
	tokens = []
	for s in sentences:
		tokenizer = RegexpTokenizer(r'\w+')
		tokens += tokenizer.tokenize(s.lower())
	tokens_filtred =[x for x in tokens if x not in Util.ENGLISH_STOP_WORDS]

	st = PorterStemmer()
	return [st.stem(x) for x in tokens_filtred]

def get_feature_jaccard_distance(tokens1, tokens2):
	return 0 if(len(tokens1) == 0 and len(tokens2) == 0) else distance.jaccard(tokens1, tokens2)

def get_feature_leveinshtein_distance(tokens1, tokens2):
	return distance.nlevenshtein(tokens1, tokens2)

def get_feature_count_equals_words(tokens1, tokens2):
	count = 0
	for item in tokens1:
		if item in tokens2:
			count += 1
	return [0 if len(tokens1) == 0 else count/len(tokens1),
			0 if len(tokens2) == 0 else count/len(tokens2)]

def generate_X(q1, q2):
	tokens1 = tokenize_question(q1)
	tokens2 = tokenize_question(q2)

	X = []
	X.append(get_feature_leveinshtein_distance(tokens1, tokens2))
	X.append(get_feature_jaccard_distance(tokens1, tokens2))
	X += get_feature_count_equals_words(tokens1, tokens2)

	return X

def generate_train_features(n_examples):
	print("*** generating train featues ***")
	X_train = []
	y_train = []

	i = 0

	with open(TRAIN_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X(line[3], line[4])
			X_train.append(x)
			y_train.append(line[5])
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1
			if(n_examples > 0 and i == n_examples): break

	with open(TRAIN_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_train):
			line = [index] + item + [int(y_train[index])]
			csv_writer.writerow(line)

def generate_test_features():
	print("*** generating test features ***")
	X_test = []

	i = 0
	with open(TEST_FILE, "r") as f:
		csv_reader = csv.reader(f, delimiter=',', quotechar='"')
		next(csv_reader) # jumping the header stuff
		for index, line in enumerate(csv_reader):
			x = generate_X(line[1], line[2])
			X_test.append(x)
			if(i % 1000 == 0): print(str(i) + '\r', end='')
			i += 1

	with open(TEST_FEATURES, 'w') as f:
		csv_writer = csv.writer(f)
		for index, item in enumerate(X_test):
			line = [index] + item
			csv_writer.writerow(line)


def generate_features(train=True, test=True, n_examples=10000):
	if(train):
		start = Util.get_time()
		generate_train_features(n_examples)
		Util.the_time(start, "generating train features")
	if(test):
		start = Util.get_time()
		generate_test_features()
		Util.the_time(start, "generating test features")