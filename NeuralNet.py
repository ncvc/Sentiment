import urllib
import datetime
import csv
import os
import logging

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from Preprocess import TS_FILENAME, Preprocess, MultiTopicWordCounterTs
from SentimentAnalysis import SentimentAnalysis
from TimeSeries import TimeSeries


DATASET_FILENAME = 'dataset.out'
STOCK_DATA_FOLDER = 'stock_data'


# Make a global logging object.
logit = logging.getLogger("logit")
logit.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s:%(lineno)d - %(message)s")

# Stream handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logit.addHandler(handler)

# File handler
fileHandler = logging.FileHandler("NeuralNet.log")
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG)
logit.addHandler(fileHandler)


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
		return tuple([ts[day - j] for j in xrange(1, 1+self.historySize) for ts in (self.inputTs, self.targetTs)])

	# Creates a SupervisedDataSet from the given score and stock timeseries
	def buildDataSet(self, filename):
		logit.info('Building dataset')
		self.ds = SupervisedDataSet(self.historySize * 2, 1)

		# Hack because for some absurd reason the stocks close on weekends
		for i in xrange(self.historySize+1, len(self.targetTs)):
			# inputs - the last historySize of score and stock data
			self.ds.addSample(self.getInput(i), (self.targetTs[i],))

		self.ds.saveToFile(filename)

	# Train the neural net on the previously generated dataset
	def train(self):
		logit.info('Creating Trainer')
		trainer = BackpropTrainer(self.untrainedNet, dataset=self.ds)

		logit.info('Training Neural Net')
		for i in xrange(100):
			logit.info('Epoch %i: error = %f' % (i, trainer.train()))
		# self.trainedNet = trainer.trainUntilConvergence()
		logit.info('Finished Training Neural Net')

	# Uses the trained neural net to predict the stock at the given day and returns (actual, predicted)
	def predict(self, day):
		if self.trainedNet == None:
			logit.warn("You haven\'t trained the network yet!")
			return

		return self.inputTs[day], self.trainedNet.activate(self.getInput(day))


# Returns a feed-forward network
def createFFNet(historySize):
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
def createRecurrentNet(historySize):
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
	def __init__(self, startDate, endDate, wordCounterTsFilename=TS_FILENAME):
		self.startDate = startDate
		self.endDate = endDate
		self.wordCounterTs = Preprocess.loadTs(wordCounterTsFilename)

	# Loads stock data from file into a timeseries (list)
	def loadStockData(self, stock):
		filepath = os.path.join(STOCK_DATA_FOLDER, stock + '[%s,%s].csv' % (self.startDate, self.endDate))

		if not os.path.exists(filepath):
			logit.info('Downloading Stock Data')

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

		return TimeSeries({ datetime.datetime.strptime(row['Date'], '%Y-%m-%d').date(): float(row['Close']) for row in csv.DictReader(open(filepath)) })

	def generateNeuralNet(self, stock):
		sent = SentimentAnalysis(logit)

		topicWordCountTs = TimeSeries(self.wordCounterTs.getTopicTs(stock))

		inputTs = sent.getScoreTimeseries(topicWordCountTs)

		targetTs = self.loadStockData(stock)

		# Scale data linearly from [0,1]
		minInput = inputTs.getMinValue()
		maxInput = inputTs.getMaxValue()
		scaledInputTs = [float(val-minInput)/(maxInput-minInput) for val in inputTs.getValueList()]
		minTarget = targetTs.getMinValue()
		maxTarget = targetTs.getMaxValue()
		scaledTargetTs = [float(val-minTarget)/(maxTarget-minTarget) for val in targetTs.getValueList()]

		nn = NeuralNet(createFFNet, 3, scaledInputTs, scaledTargetTs)
		nn.train()

		return nn


if __name__ == '__main__':
	net = StockNeuralNet(datetime.date(2011, 1, 1), datetime.date(2011, 12, 31))
	net.generateNeuralNet('goog')

	# nn = NeuralNet(createFFNet, 3, [])
	# nn.train()
	# actual, predicted = nn.predict(10)
