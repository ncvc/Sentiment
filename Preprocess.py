import cPickle as pickle
import datetime
import string
import json
import collections
import time

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from parse_hn.database import DB
from TimeSeries import TimeSeries, dateIterator


TOPICS_FILENAME = 'topics.json'
TS_FILENAME = 'ts.p'


# self.topicsWordCounterTs - a dictionary of timeseries data, where the keys are topics and the values are TimeSeries objects.
#           The values of each time series object is a collections.Counter object with the keys being words and the values being
#           the number of times that word appears in a comment on a post of the given topic.
#
#           { <topic1>: TimeSeries({ <date1>: Counter( { <word1>: <# of times word1 appears in the comments of posts categorized as <topic1>, ... } ), ... }, ... )}
#
class MultiTopicWordCounterTs:
	def __init__(self, startDate, endDate, topics):
		self.topicsWordCounterTs = {}
		for topic in topics:
			self.topicsWordCounterTs[topic] = TimeSeries({ date: collections.Counter() for date in dateIterator(startDate, endDate) })

	def addCommentCounter(self, wordCounter, date, topic):
		self.topicsWordCounterTs[topic].ts[date] += wordCounter

	def getTopicTs(self, topic):
		return self.topicsWordCounterTs[topic]


# Generates a summary of the comment data as self.multiTopicWordCounterTs
class Preprocess:
	def __init__(self, startDate, endDate, db, topicsFilename=TOPICS_FILENAME):
		stopwordsSet = set(stopwords.words('english'))
		punctuation = set(string.punctuation)
		self.ignoreWords = stopwordsSet.union(punctuation)

		self.startDate = startDate
		self.endDate = endDate

		self.loadTopics(topicsFilename)
		self.multiTopicWordCounterTs = MultiTopicWordCounterTs(startDate, endDate, self.topicKeywordDict.keys())

		self.itemTopics = {}
		self.db = db

	@classmethod
	def loadTs(self, filename=TS_FILENAME):
		return pickle.load(open(filename, 'rb'))

	def saveTs(self, filename=TS_FILENAME):
		pickle.dump(self.multiTopicWordCounterTs, open(filename, 'wb'))

	def loadTopics(self, filename):
		topics = json.load(open(filename))
		self.topicKeywordDict = { topic.lower(): [ keyword.lower() for keyword in keywords + [topic] ] for topic, keywords in topics.iteritems() }

	# Helper - count the individual words in a single comment
	def preprocessComment(self, text, mult=1):
		wordCount = collections.Counter()

		for sentence in sent_tokenize(text.lower()):
			for word in word_tokenize(sentence):
				# Don't bother with stop words and punctuation
				if word not in self.ignoreWords:
					wordCount[word] += 1 * mult

		return wordCount

	# Returns the topic list of the given post title
	def getTitleTopicList(self, title):
		if title == None:
			return []
		return [topic for topic, keywords in self.topicKeywordDict.iteritems() if any(keyword in title.lower() for keyword in keywords)]

	# Helper method to return the topics of the parent of an item
	def getParentTopics(self, item):
		parentId = item.parent

		# Return the topics of the parent if the parent has already been processed
		if parentId in self.itemTopics:
			topics = self.itemTopics[parentId]
		elif item.type == 'comment':
			topics = self.getParentTopics(self.db.get_story(parentId))
		else:
			topics = self.getTitleTopicList(item.title)

		self.itemTopics[item.id] = topics
		return topics

	# Returns a list of the topics the comment pertains to
	def getCommentTopics(self, comment):
		if comment.id in self.itemTopics:
			return self.itemTopics[comment.id]

		return self.getParentTopics(comment)

	# Preprocess the entire db and populate self.multiTopicWordCounterTs
	def preprocess(self, saveTs=True, filename=TS_FILENAME):
		print 'preprocess start'
		i = 0
		start = mid = time.clock()
		for comment in self.db.get_comments(self.startDate, self.endDate):
			i += 1
			if i % 10000 == 1:
				print i, time.clock() - mid
				mid = time.clock()
			relevantTopics = self.getCommentTopics(comment)

			# Don't bother if there are no relevantTopics
			if len(relevantTopics) > 0 and comment.score > 10:
				commentWordCounter = self.preprocessComment(comment.text, mult=comment.score)

				date = comment.time.date()
				# add the comment's word count to the relevant day's wordcount in the topic
				for topic in relevantTopics:
					self.multiTopicWordCounterTs.addCommentCounter(commentWordCounter, date, topic)


		if saveTs:
			self.saveTs(filename)

		print 'Total', time.clock() - start

		return self.multiTopicWordCounterTs

if __name__ == '__main__':
	startDate = datetime.date(2011, 1, 1)
	endDate = datetime.date(2012, 1, 1)

	with DB() as db:
		p = Preprocess(startDate, endDate, db)
		p.preprocess(filename='ts-greater-than-10-with-score-mult.p')
