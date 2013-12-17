from SentimentAnalysis import SentimentAnalysis
from Preprocess import TS_FILENAME, Preprocess, MultiTopicWordCounterTs


wordCounters = {}
sent = SentimentAnalysis()


def loadScoreTs(stock, wordCounterTsFilename=TS_FILENAME):
	if wordCounterTsFilename not in wordCounters:
		wordCounters[wordCounterTsFilename] = Preprocess.loadTs(wordCounterTsFilename)
	wordCounterTs = wordCounters[wordCounterTsFilename]

	topicWordCountTs = wordCounterTs.getTopicTs(stock)
	return sent.getScoreTimeseries(topicWordCountTs)
