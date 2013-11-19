import string
import collections
import cPickle as pickle

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import POS_LIST
from nltk.tokenize import word_tokenize, sent_tokenize


WORDLIST_FILENAME = 'subjectivity_clues/subjclueslen1-HLTEMNLP05.tff'
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


# Generates a summary of the comment data as self.topics
# self.topics - a dictionary of timeseries data, where the keys are topic words, e.g. 'apple', 'microsoft', etc.
#               the values are lists, one element for each day of the year. Each element of the list is a collections.Counter
#               object with the keys being words and the values being the number of times that word appears in a comment
#               on a post of the given topic.
#               { <topic>: [ Counter( { 'computer': 254, 'java': 24, ... } ), ... ], ... }
class Preprocess:
	def __init__(self, topicList):
		stopwordsSet = set(stopwords.words('english'))
		punctuation = set(string.punctuation)
		self.ignoreWords = stopwordsSet.union(punctuation)

		self.topicList = [topic.lower() for topic in topicList]
		self.topics = { topic: [collections.Counter() for i in xrange(365)] for topic in self.topicList }

	@staticmethod
	def loadWordCounts(filename):
		return pickle.load(open(filename, 'rb'))

	def saveWordCounts(self, filename):
		pickle.dump(self.topics, open(filename, 'wb'))

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
		return [topic for topic in self.topicList if topic in post.title.lower()]

	def getPosts(self):
		return ['posts']

	def getComments(self, post):
		return ['comments']

	# Preprocess the entire db and populate self.topics
	def preprocess(self):
		for post in self.getPosts():
			topics = self.getPostTopics(post)

			# Don't bother if there are no relevant topics
			if len(topics) > 0:
				topicWordCounts = [self.topics[topic] for topic in topics]
				for comment in self.getComments(post):
					text = comment.text
					day = comment.day
					commentWordCount = self.preprocessComment(text)

					# add the comment's word count to the relevant day's wordcount in the topic
					for topicWordCount in topicWordCounts:
						topicWordCount[day] += commentWordCount


# Uses the parsed wordlist to get a timeseries of sentiment scores
class SentimentAnalysis:
	def __init__(self, wordCounts):
		self.wl = Wordlist()

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


if __name__ == '__main__':
	topicList = ['apple', 'microsoft', 'facebook', 'twitter', 'dell', 'intel', 'bitcoin']

	preproc = Preprocess(topicList)
	preproc.preprocess()
	preproc.saveWordCounts('wordcount.p')
	wordCounts = preproc.wordCounts
	# wordCounts = Preprocess.loadWordCounts('wordcount.p')

	sent = SentimentAnalysis()
	ts = sent.getScoreTimeseries(wordCounts)
