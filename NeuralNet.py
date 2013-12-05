import urllib
import datetime
import csv
import os
import logging
import itertools
import random

from pybrain.structure           import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

# File handler
fileHandler = logging.FileHandler("NeuralNet-data.log")
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.INFO)
logit.addHandler(fileHandler)


# Given a score timeseries and function to create a neural net, train the neural net to predict a target timeseries.
# The past historySize inputs and targets are used as inputs to the neural net
class NeuralNet:
	def __init__(self, createNN, historySize, scoreTs, targetTs, targetScalingFactor=1.0, targetOffset=0.0, testProportion=0.25, loadDataSetFromFile=False, dataSetFilename=DATASET_FILENAME):
		self.historySize = historySize
		self.scoreTs = scoreTs
		self.targetTs = targetTs

		self.net = createNN(self.historySize)

		self.testProportion = testProportion
		self.targetScalingFactor = targetScalingFactor
		self.targetOffset = targetOffset

		if loadDataSetFromFile:
			self.loadDataSets(dataSetFilename)
		else:
			self.buildDataSets(dataSetFilename)

	# Load a DataSet from the given file
	def loadDataSets(self, filename):
		self.testDs = SupervisedDataSet.loadFromFile('test' + filename)
		self.trainDs = SupervisedDataSet.loadFromFile('train' + filename)

	# Convenience method to return the inputs for a given day
	def getInputs(self, date):
		timedelta = datetime.timedelta(1)
		return tuple(itertools.chain.from_iterable((ts.pastNVals(date - timedelta, self.historySize) for ts in (self.scoreTs, self.targetTs))))

	def getTrainAndTestDateLists(self):
		fullDateList = self.targetTs.getDateList()
		numTrainingDates = int(len(fullDateList) * (1.0-self.testProportion))

		return fullDateList[self.historySize+1:numTrainingDates], fullDateList[numTrainingDates:]

	def buildDataSets(self, filename):
		trainDateList, testDateList = self.getTrainAndTestDateLists()
		logit.debug('Building training dataset')
		self.trainDs = self.buildDataSet('train' + filename, trainDateList)
		logit.debug('Building test dataset')
		self.testDs = self.buildDataSet('test' + filename, testDateList)

	# Creates a SupervisedDataSet from the given score and stock timeseries
	def buildDataSet(self, filename, dateList):
		ds = SupervisedDataSet(self.historySize*2, 1)

		# Hack because for some absurd reason the stocks close on weekends
		for date in dateList:
			# inputs - the last historySize of score and stock data
			ds.addSample(self.getInputs(date), (self.targetTs.getVal(date),))

		ds.saveToFile(filename)

		return ds

	# Train the neural net on the previously generated dataset
	def train(self):
		logit.debug("Number of training patterns: %i" % len(self.trainDs))
		logit.debug("Input and output dimensions: %i, %i" % (self.trainDs.indim, self.trainDs.outdim))
		logit.debug("First sample (input, target):")
		logit.debug(str((self.trainDs['input'][0], self.trainDs['target'][0])))
		
		logit.debug('Creating Trainer')
		trainer = BackpropTrainer(self.net, dataset=self.trainDs, verbose=True)

		logit.debug('Training Neural Net')
		trainer.trainUntilConvergence(maxEpochs=1000)
		logit.debug('Finished Training Neural Net')
		return self.net

	def unscale(self, val):
		return self.targetOffset + val / self.targetScalingFactor

	def predict(self, inputs):
		return self.unscale(self.net.activate(inputs)[0])

	# Uses the trained neural net to predict the stock in the given date range and returns a list of predicted values
	def predictTs(self, dateList):
		return TimeSeries({ date: self.predict(self.getInputs(date)) for date in dateList })

	def plotResult(self):
		targetX, targetY = zip(*self.targetTs.getItems())
		targetY = [self.unscale(y) for y in targetY]
		actualTargetTs = self.targetTs.mapValues(self.unscale)

		trainX, testX = self.getTrainAndTestDateLists()

		# Configure plots
		fig, ax = plt.subplots()
		y_formatter = ScalarFormatter(useOffset=False)
		ax.yaxis.set_major_formatter(y_formatter)
		plt.subplot(211)
		plt.plot(targetX, targetY, 'r', label='Targets')
		plt.legend()

		for x, color, label in ((trainX, 'g', 'Training'), (testX, 'b', 'Testing')):
			predictedTs = self.predictTs(x)

			error = [float(predictedTs.getVal(date) - actualTargetTs.getVal(date)) for date in x]

			mse = sum((e**2 for e in error)) / len(x)
			logit.info('%s MSE: %f' % (label, mse))

			mape = sum([abs(float(predictedTs.getVal(date) - actualTargetTs.getVal(date)) / actualTargetTs.getVal(date)) for date in x]) / len(x) * 100.0
			logit.info('%s MAPE: %f' % (label, mape))

			correct = 0
			for date in x[1:]:
				today, yesterday = predictedTs.pastNVals(date, 2)
				todayTarget, yesterdayTarget = actualTargetTs.pastNVals(date, 2)
				if (today - yesterdayTarget) * (todayTarget - yesterdayTarget) > 0:
					correct += 1
			dirAcc = float(correct) / (len(x) - 1)
			logit.info('%s Directional Accuracy from actual: %f' % (label, dirAcc))

			correct = 0
			for date in x[1:]:
				today, yesterday = predictedTs.pastNVals(date, 2)
				todayTarget, yesterdayTarget = actualTargetTs.pastNVals(date, 2)
				if (today - yesterday) * (todayTarget - yesterdayTarget) > 0:
					correct += 1
			dirAcc = float(correct) / (len(x) - 1)
			logit.info('%s Directional Accuracy from prediction: %f' % (label, dirAcc))

			# Prediction plot
			plt.subplot(211)
			plt.plot(x, predictedTs.getValueList(), color, label=label)

			# Squared Error plot
			plt.subplot(212)
			plt.fill_between(x, error, facecolor=color)


# Returns a feed-forward network
def createFFNet(historySize):
	net = FeedForwardNetwork()

	# Create and add layers
	net.addInputModule(LinearLayer(historySize * 2, name='in'))
	net.addModule(SigmoidLayer(10, name='hidden'))
	net.addOutputModule(LinearLayer(1, name='out'))

	# Create and add connections between the layers
	net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
	net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))

	# Preps the net for use
	net.sortModules()

	return net


# Returns a feed-forward network with n layers of k neurons each
def createNLayerFFNet(historySize, n, k):
	net = FeedForwardNetwork()

	# Create and add layers
	net.addInputModule(LinearLayer(historySize * 2, name='in'))
	net.addOutputModule(LinearLayer(1, name='out'))

	# Create and add connections between the layers
	baseLayerName = 'hidden%i'
	connectionName = 'c%i'

	net.addModule(SigmoidLayer(k, name=baseLayerName % 0))
	net.addConnection(FullConnection(net['in'], net[baseLayerName % 0], name=connectionName % 0))
	
	for i in xrange(1, n):
		layerName = baseLayerName % i
		inLayerName = baseLayerName % (i-1)

		net.addModule(SigmoidLayer(k, name=layerName))
		net.addConnection(FullConnection(net[inLayerName], net[layerName], name=connectionName % (i-1)))

	net.addConnection(FullConnection(net[baseLayerName % (n-1)], net['out'], name=connectionName % (n-1)))

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
		self.sent = SentimentAnalysis(logit)
		self.scores = {}

	def loadScoreTs(self, stock):
		if stock not in self.scores:
			topicWordCountTs = self.wordCounterTs.getTopicTs(stock)
			self.scores[stock] = self.sent.getScoreTimeseries(topicWordCountTs)

		return self.scores[stock]

	# Loads stock data from file into a timeseries (list)
	def loadStockData(self, stock):
		filepath = os.path.join(STOCK_DATA_FOLDER, stock + '[%s,%s].csv' % (self.startDate, self.endDate))

		if not os.path.exists(filepath):
			logit.debug('Downloading Stock Data')

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

	def generateNeuralNet(self, stock, loadDataSetFromFile=False):
		scoreTs = self.loadScoreTs(stock)
		targetTs = self.loadStockData(stock)

		# Scale data linearly from [0,1]
		minInput = scoreTs.getMinValue()
		maxInput = scoreTs.getMaxValue()
		try:
			inpScale = 1.0 / (maxInput-minInput)
		except ZeroDivisionError:
			inpScale = 1
		scaledInputTs = scoreTs.mapValues(lambda val: float(val-minInput) * inpScale)

		minTarget = targetTs.getMinValue()
		maxTarget = targetTs.getMaxValue()
		targetScale = 1.0 / (maxTarget-minTarget)
		scaledTargetTs = targetTs.mapValues(lambda val: float(val-minTarget) * targetScale)

		createNet = lambda historySize: createNLayerFFNet(historySize, 5, 10)
		nn = NeuralNet(createNet, 3, scaledInputTs, scaledTargetTs, targetScalingFactor=targetScale, targetOffset=minTarget, loadDataSetFromFile=loadDataSetFromFile)
		nn.train()

		return nn


if __name__ == '__main__':
	stocks = ['intc', 'aapl', 'msft', 'goog', 'dell'] # , 'fb', 'twtr'
	files = ['ts-greater-than-10-score.p', 'ts-greater-than-100-score.p', 'ts-greater-than-50-score.p', 'ts-pg-only-with-score-mult.p', 'ts-pg-only.p', 'ts-pos-score.p', 'ts-with-score-multiplier.p', 'ts.p']
	for filename in files:
		for stock in stocks:
			logit.info('file: %s' % filename)
			logit.info('stock: %s' % stock)
			net = StockNeuralNet(datetime.date(2011, 1, 1), datetime.date(2011, 12, 31), wordCounterTsFilename=filename)
			nn = net.generateNeuralNet(stock, loadDataSetFromFile=False)
			nn.plotResult()

			fig = plt.gcf()
			fig.canvas.set_window_title('stock %s, file %s' % (stock, filename))
			plt.draw()

	plt.show()

	# nn = NeuralNet(createFFNet, 3, [])
	# nn.train()
	# actual, predicted = nn.predict(10)
