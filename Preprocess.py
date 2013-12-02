import cPickle as pickle
import datetime
import string
import json
import collections
import time

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from parse_hn.database import DB


TOPICS_FILENAME = 'topics.json'
TS_FILENAME = 'ts.p'


# Generates a summary of the comment data as self.topicsWordCounterTs
# self.topicsWordCounterTs - a dictionary of timeseries data, where the keys are datetime.date objects, one for each day in the
#           range [startDate, endDate]. The values are dictionaries with keys that are topic words, e.g. 'apple', 'microsoft', etc.
#           The values are lists, where each element of the list is a collections.Counter object with the keys being words
#           and the values being the number of times that word appears in a comment on a post of the given topic.
#
#           { <date1>: { <topic1>: [ Counter( { <word1>: <# of times word1 appears in the comments of posts categorized as <topic1>, ... } ), ... ], ... }, ... }
#
class Preprocess:
	def __init__(self, startDate, endDate, db, topicsFilename=TOPICS_FILENAME):
		stopwordsSet = set(stopwords.words('english'))
		punctuation = set(string.punctuation)
		self.ignoreWords = stopwordsSet.union(punctuation)

		self.startDate = startDate
		self.endDate = endDate

		self.loadTopics(topicsFilename)

		self.itemTopics = {}
		self.db = db

	@classmethod
	def loadTs(self, filename=TS_FILENAME):
		return pickle.load(open(filename, 'rb'))

	def saveTs(self, filename=TS_FILENAME):
		pickle.dump(self.topicsWordCounterTs, open(filename, 'wb'))

	def loadTopics(self, filename):
		topics = json.load(open(filename))
		self.topicKeywordDict = { topic.lower(): [ keyword.lower() for keyword in keywords + [topic] ] for topic, keywords in topics.iteritems() }
		self.topicsWordCounterTs = {}
		for day in xrange(self.startDate.toordinal(), self.endDate.toordinal()+1):
			self.topicsWordCounterTs[datetime.date.fromordinal(day)] = { topic: collections.Counter() for topic in self.topicKeywordDict.iterkeys() }

	# Helper - count the individual words in a single comment
	def preprocessComment(self, text):
		wordCount = collections.Counter()

		for sentence in sent_tokenize(text.lower()):
			for word in word_tokenize(sentence):
				# Don't bother with stop words and punctuation
				if word not in self.ignoreWords:
					wordCount[word] += 1

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

	# Preprocess the entire db and populate self.topicsWordCounterTs
	def preprocess(self, saveTs=True, filename=TS_FILENAME):
		print 'preprocess start'
		i = 0
		start = mid = time.time()
		for comment in self.db.get_comments(self.startDate, self.endDate):
			i += 1
			if i % 10000 == 1:
				print i, time.time() - mid
				mid = time.time()
			relevantTopics = self.getCommentTopics(comment)

			# Don't bother if there are no relevantTopics
			if len(relevantTopics) > 0:
				commentWordCounter = self.preprocessComment(comment.text)

				date = comment.time.date()
				# add the comment's word count to the relevant day's wordcount in the topic
				for topic in relevantTopics:
					self.topicsWordCounterTs[date][topic] += commentWordCounter

		if saveTs:
			self.saveTs(filename)

		print 'Total', time.time() - start

		return self.topicsWordCounterTs

if __name__ == '__main__':
	startDate = datetime.datetime(2011, 1, 1)
	endDate = datetime.datetime(2011, 12, 31)

	with DB() as db:
		p = Preprocess(startDate, endDate, db)
		p.preprocess()
