import string
import collections
import cPickle as pickle
import json

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import POS_LIST
from nltk.tokenize import word_tokenize, sent_tokenize

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

WORDLIST_FILENAME = 'subjectivity_clues/subjclueslen1-HLTEMNLP05.tff'
STOCK_DATA_FILENAME = 'stocks.whatever'
DATASET_FILENAME = 'dataset.out'
TOPICS_FILENAME = 'topics.json'
WORDCOUNT_FILENAME = 'wordcount.p'
POLARITY = { 'positive': 0, 'negative': 1, 'both': 2, 'neutral': 3 }


# Handles the subjectivity wordlist
class Wordlist:
	def __init__(self, filename=WORDLIST_FILENAME):
		self.stemmedWordlist = {}
		self.unstemmedWordlist = {}

		self.lemmatizer = WordNetLemmatizer()

		self.loadWordlist(filename)

	# Load the wordlist
	def loadWordlist(self, filename):
		for line in open(filename):
			fields = line.strip().split(' ')

			# Parse into a dict
			fieldDict = {}
			for field in fields:
				name, value = field.split('=')
				fieldDict[name] = value

			# Parse dict
			isStrong = (fieldDict['type'] == 'strongsubj')   # Strongly or weakly subjective
			# wordDict['pos'] = fieldDict['pos1']                       # Part of speech (Don't care for now)
			isStemmed = (fieldDict['stemmed1'] == 'y')       # Is the word stemmed?
			polarity = POLARITY[fieldDict['priorpolarity']]  # Prior polarity of the word

			self.addWord(fieldDict['word1'], isStemmed, isStrong, polarity)

	# Adds the word to the wordlist
	def addWord(self, word, isStemmed, isStrong, polarity):
		wordDict = { 'isStrong': isStrong, 'polarity': polarity }

		if isStemmed:
			wordlist = self.stemmedWordlist
		else:
			wordlist = self.unstemmedWordlist

		if word in wordlist:
			if wordlist[word] == None or wordDict['polarity'] != wordlist[word]['polarity']:
				# print 'COLLISION! %s: %s, %s' % (word, wordDict, wordlist[word])
				wordlist[word] = None
		else:
			wordlist[word] = wordDict

	# Returns the properties of the given word
	def getWordDict(self, word):
		wordDict = None

		# First, try just the word and use that if we can
		if word in self.unstemmedWordlist:
			wordDict = self.unstemmedWordlist[word]
		else: # Otherwise try all possible word stems
			for pos in POS_LIST:
				stemmedWord = self.lemmatizer.lemmatize(word, pos)
				if stemmedWord in self.stemmedWordlist:
					wordDict = self.stemmedWordlist[stemmedWord]
					break

		return wordDict


# Generates a summary of the comment data as self.topicsWordCount
# self.topicsWordCount - a dictionary of timeseries data, where the keys are topic words, e.g. 'apple', 'microsoft', etc.
#               the values are lists, one element for each day of the year. Each element of the list is a collections.Counter
#               object with the keys being words and the values being the number of times that word appears in a comment
#               on a post of the given topic.
#               { <topic>: [ Counter( { 'computer': 254, 'java': 24, ... } ), ... ], ... }
class Preprocess:
	def __init__(self, topicsFilename=TOPICS_FILENAME):
		stopwordsSet = set(stopwords.words('english'))
		punctuation = set(string.punctuation)
		self.ignoreWords = stopwordsSet.union(punctuation)

		self.loadTopics(topicsFilename)

	@staticmethod
	def loadWordCounts(filename):
		return pickle.load(open(filename, 'rb'))

	def saveWordCounts(self, filename):
		pickle.dump((self.topicsWordCount, self.topicList), open(filename, 'wb'))

	def loadTopics(self, filename):
		topics = json.load(open(filename))
		self.topicList = { topic.lower(): [ keyword.lower() for keyword in keywords + [topic] ] for topic, keywords in topics.iteritems() }
		self.topicsWordCount = { topic: [collections.Counter() for i in xrange(365)] for topic in self.topicList.iterkeys() }

	# Helper - count the words in a single comments
	def preprocessComment(self, text):
		wordCount = collections.Counter()

		for sentence in sent_tokenize(text):
			for word in word_tokenize(sentence):
				# Don't bother with stop words and punctuation
				if word not in self.ignoreWords:
					wordCount[word.decode('utf8')] += 1

		return wordCount

	# Returns a list of the topics the post pertains to
	def getPostTopics(self, post):
		return [topic for topic, keywords in self.topicList.iteritems() if any(keyword in post.title.lower() for keyword in keywords)]

	def getPosts(self):
		return ['posts']

	def getComments(self, post):
		return ['comments']

	# Preprocess the entire db and populate self.topicsWordCount
	def preprocess(self):
		for post in self.getPosts():
			relevantTopics = self.getPostTopics(post)

			# Don't bother if there are no relevantTopics
			if len(relevantTopics) > 0:
				relevantTopicsWordCount = [self.topicsWordCount[topic] for topic in relevantTopics]
				for comment in self.getComments(post):
					text = comment.text
					day = comment.day
					commentWordCount = self.preprocessComment(text)

					# add the comment's word count to the relevant day's wordcount in the topic
					for relevantTopicWordCount in relevantTopicsWordCount:
						relevantTopicWordCount[day] += commentWordCount

		return self.topicsWordCount


# Uses the parsed wordlist to get a timeseries of sentiment scores
class SentimentAnalysis:
	def __init__(self):
		self.wl = Wordlist()

	# Returns the sentiment score for a given day
	def getDayScore(self, wordCount):
		positiveScore = 0
		negativeScore = 0
		for word, count in self.wordCount.iteritems():
			wordDict = self.wl.getWordDict(word)

			if wordDict['polarity'] == POLARITY['positive']:
				positiveScore += count
			elif wordDict['polarity'] == POLARITY['negative']:
				negativeScore += count

		return float(positiveScore) / negativeScore

	# Return the score timeseries as a list
	def getScoreTimeseries(self, wordCounts):
		return [self.getDayScore(wordCount) for wordCount in wordCounts]


# Given a score timeseries and function to create a neural net, train the neural net to predict stocks
class StockNeuralNet:
	def __init__(self, createNN, numInputDays, scoreTs, loadDataSetFromFile=False, stockDataFilename=STOCK_DATA_FILENAME, dataSetFilename=DATASET_FILENAME):
		self.trainedNet = None
		self.numInputDays = numInputDays
		self.scoreTs = scoreTs

		self.untrainedNet = createNN(self.numInputDays)

		self.loadStockData(stockDataFilename)

		if loadDataSetFromFile:
			self.loadDataSet(dataSetFilename)
		else:
			self.buildDataSet(dataSetFilename)

	# Loads stock data from file into a timeseries (list)
	def loadStockData(self, filename):
		self.stockTs = []

	# Load a DataSet from the given file
	def loadDataSet(self, filename):
		self.ds = SupervisedDataSet.loadFromFile(filename)

	# Convenience method to return the inputs for a given day
	def getInput(self, day):
		return tuple([self.scoreTs[day - j] for j in xrange(1, 1+self.numInputDays)] + [self.stockTs[day - j] for j in xrange(1, 1+self.numInputDays)])

	# Creates a SupervisedDataSet from the given score and stock timeseries
	def buildDataSet(self, filename):
		if len(self.scoreTs) != len(self.stockTs):
			print 'scoreTs != stockTs!!!!!!!!'

		self.ds = SupervisedDataSet(self.numInputDays * 2, 1)

		for i in xrange(self.numInputDays, len(self.scoreTs)):
			# inputs - the last numInputDays of score and stock data
			self.ds.addSample(self.getInput(i), (self.stockTs[i],))

		self.ds.saveToFile(filename)

	# Train the neural net on the previously generated dataset
	def train(self):
		trainer = BackpropTrainer(self.untrainedNet, self.ds)

		self.trainedNet = trainer.trainUntilConvergence()

	# Uses the trained neural net to predict the stock at the given day and returns (actual, predicted)
	def predict(self, day):
		if self.trainedNet == None:
			print 'You haven\'t trained the network yet!'
			return

		return self.scoreTs[day], self.trainedNet.activate(getInput(day))



# Returns a feed-forward network
def createFFNet(numInputDays):
	net = FeedForwardNetwork()

	# Create and add layers
	net.addInputModule(LinearLayer(numInputDays * 2, name='in'))
	net.addModule(SigmoidLayer(5, name='hidden'))
	net.addOutputModule(LinearLayer(1, name='out'))

	# Create and add connections between the layers
	net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
	net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))

	# Preps the net for use
	net.sortModules()

	return net

# Returns a recurrent network
def createRecurrentNet(numInputDays):
	net = RecurrentNetwork()

	# Create and add layers	
	net.addInputModule(LinearLayer(numInputDays * 2, name='in'))
	net.addModule(SigmoidLayer(5, name='hidden'))
	net.addOutputModule(LinearLayer(1, name='out'))

	# Create and add connections between the layers
	net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
	net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))
	net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))

	# Preps the net for use
	net.sortModules()

	return net


def generateTopicsWordCounts():
	preproc = Preprocess()
	topicsWordCount = preproc.preprocess()
	preproc.saveWordCounts(WORDCOUNT_FILENAME)
	return topicsWordCount


def loadWordCounts():
	return Preprocess.loadWordCounts(WORDCOUNT_FILENAME)


def getTopicsNN():
	topicsWordCount = generateWordCounts()
	# topicsWordCount, topicList = loadWordCounts()

	topicsNN = {}
	sent = SentimentAnalysis()
	for topic, topicWordCount in topicsWordCount.iteritems():
		scoreTs = sent.getScoreTimeseries(topicWordCount)

		nn = StockNeuralNet(createFFNet, 3, scoreTs)
		nn.train()

		topicsNN[topic] = nn

	return topicsNN


if __name__ == '__main__':
	# topicsNN = getTopicsNN()

	nn = StockNeuralNet(createFFNet, 3, [])
	nn.train()
	actual, predicted = nn.predict(10)
