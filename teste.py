import csv
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as ax
from features_generator import *
from util import *
from sklearn.feature_extraction.text import CountVectorizer
import operator
from nltk.corpus import stopwords

# X = []
# y = []
# plt.scatter(X)
# Sigma = SVD.explained_variance_ratio_
# VT = SVD.components_
# print(Sigma)

# plt.scatter(range(len(Sigma)), Sigma)
# plt.show()

# with open('files/train_tokenized.csv', 'r') as f:
# 	csv_reader = csv.reader(f, delimiter=',', quotechar='"')
# 	distinct = {}
# 	i = 0
# 	for record in csv_reader:
# 		t1 = record[3]
# 		t2 = record[4]
# 		for t in [t1, t2]:
# 			for a in t.split(' '):
# 				if a not in distinct: distinct[a] = True
# 		i += 1
# 		if(i % 1000 == 0): print(str(i)+'\r',end='')
# 	print(len(distinct))

# lst = [ 0.00555549,0.00369336,0.00372798,0.0034161, 0.00333593,0.00274764
# ,0.00276581,0.00251675,0.00251888,0.00244803,0.00218998,0.00215903
# ,0.00206805,0.00204538,0.00202209,0.00181568,0.00177331,0.00173342
# ,0.00168534,0.00167697,0.00162353,0.00156997,0.00150204,0.00148863
# ,0.00148842,0.00143211,0.00140567,0.00138968,0.00137252,0.00133281
# ,0.0013098, 0.00125556,0.00122489,0.00116063,0.00112621,0.0011171
# ,0.00110469,0.00110084,0.00107797,0.001077,0.00104867,0.00102735
# ,0.00101908,0.00100047,0.00096779,0.00095233,0.00094038,0.00091944
# ,0.00090695,0.00090163]
# print(len(lst))
# plt.plot(lst)
# plt.show()


# y = [68.2416, 68.1317]
# N = len(y)
# x = range(N)
# width = 1/1.5
# plt.bar(['a','b'], y, width, color="blue")

# plt.show()
# fig = plt.gcf()
# plot_url = py.plot_mpl(fig, filename='mpl-basic-bar')



with open('files/train.csv', 'r') as f:
	corpus1 = []
	corpus2 = []
	csv_reader = csv.reader(f, delimiter=',', quotechar='"')
	for record in csv_reader:
		corpus1.append(record[3])
		corpus2.append(record[4])

	vectorizer = CountVectorizer(strip_accents='ascii',binary=True)
	tok = vectorizer.build_tokenizer()
	count = {}

	for c in corpus1:
		for w in tok(c):
			w = w.lower()
			if w not in count:
				count[w] = 1
			else:
				count[w] += 1
	sortedw = sorted(count.items(), key=operator.itemgetter(1))
	sortedw.reverse()
	# i=0
	# for s in sortedw:
		# print(s)
		# i += 1
		# if(i == 100): break


corte = sortedw[:1000]
stoplist = stopwords.words('english')
i=0
for p in corte:
	if p[0] in stoplist:
		if(p[1] <= 1000):
			print(p, 'index: ' + str(i))
	i += 1

print(len(sortedw)) # 67841 palavras diferentes
