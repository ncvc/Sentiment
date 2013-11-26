import string
import collections
import cPickle as pickle
import json
import urllib
import datetime
import csv
import os


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import POS_LIST
from nltk.tokenize import word_tokenize, sent_tokenize

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

WORDLIST_FILENAME = 'subjectivity_clues/subjclueslen1-HLTEMNLP05.tff'
DATASET_FILENAME = 'dataset.out'
TOPICS_FILENAME = 'topics.json'
WORDCOUNT_FILENAME = 'wordcount.p'
STOCK_DATA_FOLDER = 'stock_data'

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
	def __init__(self, startDate, endDate, topicsFilename=TOPICS_FILENAME):
		stopwordsSet = set(stopwords.words('english'))
		punctuation = set(string.punctuation)
		self.ignoreWords = stopwordsSet.union(punctuation)

		self.loadTopics(topicsFilename)

		self.startDate = startDate
		self.endDate = endDate

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

	# Returns all posts in the date range
	def getPosts(self):
		return ['posts']

	# Returns all comments for the given post
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


# Given a score timeseries and function to create a neural net, train the neural net to predict a target timeseries.
# The past historySize inputs and targets are used as inputs to the neural net
class NeuralNet:
	def __init__(self, createNN, historySize, inputTs, targetTs, loadDataSetFromFile=False, dataSetFilename=DATASET_FILENAME):
		self.trainedNet = None
		self.historySize = historySize
		self.inputTs = inputTs
		self.targetTs = targetTs

		self.untrainedNet = createNN(self.historySize)

		if loadDataSetFromFile:
			self.loadDataSet(dataSetFilename)
		else:
			self.buildDataSet(dataSetFilename)

	# Load a DataSet from the given file
	def loadDataSet(self, filename):
		self.ds = SupervisedDataSet.loadFromFile(filename)

	# Convenience method to return the inputs for a given day
	def getInput(self, day):
		return tuple([self.inputTs[day - j] for j in xrange(1, 1+self.historySize)] + [self.targetTs[day - j] for j in xrange(1, 1+self.historySize)])

	# Creates a SupervisedDataSet from the given score and stock timeseries
	def buildDataSet(self, filename):
		self.ds = SupervisedDataSet(self.historySize * 2, 1)

		# Hack because for some absurd reason the stocks close on weekends
		for i in xrange(self.historySize, len(self.targetTs)):
			# inputs - the last historySize of score and stock data
			self.ds.addSample(self.getInput(i), (self.targetTs[i],))

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

		return self.inputTs[day], self.trainedNet.activate(getInput(day))

	# Returns a feed-forward network
	def createFFNet(self, historySize):
		net = FeedForwardNetwork()

		# Create and add layers
		net.addInputModule(LinearLayer(historySize * 2, name='in'))
		net.addModule(SigmoidLayer(5, name='hidden'))
		net.addOutputModule(LinearLayer(1, name='out'))

		# Create and add connections between the layers
		net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
		net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))

		# Preps the net for use
		net.sortModules()

		return net

	# Returns a recurrent network
	def createRecurrentNet(self, historySize):
		net = RecurrentNetwork()

		# Create and add layers	
		net.addInputModule(LinearLayer(historySize * 2, name='in'))
		net.addModule(SigmoidLayer(5, name='hidden'))
		net.addOutputModule(LinearLayer(1, name='out'))

		# Create and add connections between the layers
		net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
		net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))
		net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))

		# Preps the net for use
		net.sortModules()

		return net


# Puts everything together and downloads stock data, analyzes sentiment, and generates neural nets for the given stocks
class StockNeuralNet:
	def __init__(self, startDate, endDate, wordCountFilename=WORDCOUNT_FILENAME):
		self.startDate = startDate
		self.endDate = endDate
		self.wordCountFilename = wordCountFilename

	# Loads stock data from file into a timeseries (list)
	def loadStockData(self, stock):
		filepath = os.path.join(STOCK_DATA_FOLDER, stock + '[%s,%s].csv' % (self.startDate, self.endDate))

		if not os.path.exists(filepath):
			if not os.path.exists(STOCK_DATA_FOLDER):
				os.makedirs(STOCK_DATA_FOLDER)

			params = urllib.urlencode({         # Below is literally the worst API design. Classic Yahoo.
				's': stock,
				'a': self.startDate.month - 1,  # WHY
				'b': self.startDate.day,
				'c': self.startDate.year,
				'd': self.endDate.month - 1,    # WHAT IS WRONG WITH YOU, YAHOO
				'e': self.endDate.day,
				'f': self.endDate.year,
				'g': 'd',
				'ignore': '.csv',               # WHAT COULD THIS POSSIBLY MEAN
			})
			url = 'http://ichart.yahoo.com/table.csv?%s' % params
			try:
				urllib.urlretrieve(url, filepath)
			except urllib.ContentTooShortError as e:
				outfile = open(filepath, "w")
				outfile.write(e.content)
				outfile.close()

				print 'Error retrieving stock %s' % stock
				return

		return [row['Close'] for row in csv.DictReader(open(filepath))]

	def loadPreprocess(self):
		return pickle.load(open(self.wordCountFilename, 'rb'))

	def savePreprocess(self, preproc):
		pickle.dump(preproc, open(self.wordCountFilename, 'wb'))

	def generateNeuralNets(self, loadPreprocessFromFile=True):
		if loadPreprocessFromFile:
			preproc = self.loadPreprocess()
		else:
			preproc = Preprocess(self.startDate, self.endDate)
			preproc.preprocess()
			self.savePreprocess(preproc)

		topicsNN = {}
		sent = SentimentAnalysis()
		for stock, topicWordCount in preproc.topicsWordCount.iteritems():
			inputTs = sent.getScoreTimeseries(topicWordCount)
			targetTs = self.loadStockData(stock)

			# Scale data linearly from [0,1]
			minInput = min(inputTs)
			maxInput = max(inputTs)
			scaledInputTs = [float(val-minInput)/(maxInput-minInput) for val in inputTs]
			minTarget = min(targetTs)
			maxTarget = max(targetTs)
			scaledTargetTs = [float(val-minTarget)/(maxTarget-minTarget) for val in targetTs]

			nn = NeuralNet(createFFNet, 3, scaledInputTs, scaledTargetTs)
			nn.train()
			topicsNN[stock] = nn

		return topicsNN


if __name__ == '__main__':
	print 'FIX TIMESERIES'
	net = StockNeuralNet(datetime.date(2012, 1, 1), datetime.date(2013, 12, 31))
	net.loadStockData('goog')
	# net.generateNeuralNets()

	# nn = NeuralNet(createFFNet, 3, [])
	# nn.train()
	# actual, predicted = nn.predict(10)
