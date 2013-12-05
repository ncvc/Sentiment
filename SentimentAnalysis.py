import time

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import POS_LIST

from Preprocess import Preprocess, MultiTopicWordCounterTs
from TimeSeries import TimeSeries


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


# Uses the parsed wordlist to get a timeseries of sentiment scores
class SentimentAnalysis:
	def __init__(self, logit):
		logit.debug('Sentiment Init')
		self.wl = Wordlist()

		self.logit = logit

	# Returns the sentiment score for a given day
	def getDayScore(self, wordCount):
		positiveScore = 0
		negativeScore = 0
		for word, count in wordCount.iteritems():
			# print word.encode(ignore), count
			wordDict = self.wl.getWordDict(word)

			if wordDict != None:
				if wordDict['polarity'] == POLARITY['positive']:
					positiveScore += count
				elif wordDict['polarity'] == POLARITY['negative']:
					negativeScore += count

		if negativeScore == 0:
			return float(positiveScore)

		return float(positiveScore) / negativeScore

	# Return the score timeseries as a list
	def getScoreTimeseries(self, wordCounterTs):
		self.logit.debug('Analyzing Sentiment')
		return wordCounterTs.mapValues(self.getDayScore)


if __name__ == '__main__':
	ts = Preprocess.loadTs()

	sent = SentimentAnalysis()
	t=time.clock()
	a = sent.getScoreTimeseries(TimeSeries(ts.getTopicTs('msft')))
	print time.clock() - t
