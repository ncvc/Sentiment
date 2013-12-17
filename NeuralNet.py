import datetime
import logging
import itertools
import random
import cPickle as pickle

from pybrain.structure           import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from Preprocess import TS_FILENAME, MultiTopicWordCounterTs
from TimeSeries import TimeSeries
from StockData import loadStockTs
from ScoreData import loadScoreTs


NN_FILENAME       = 'nn.p'
DATASET_FILENAME  = 'dataset.out'


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

	# Transforms the given scoreTs in accordance with the scale used to train this nn
	def prepareScoreTs(self, scoreTs):
		return scoreTs.linearTranform(self.targetOffset, self.targetScalingFactor)

	# Load a DataSet from the given file
	def loadDataSets(self, filename):
		self.testDs = SupervisedDataSet.loadFromFile('test' + filename)
		self.trainDs = SupervisedDataSet.loadFromFile('train' + filename)

	# Convenience method to return the inputs for a given day
	def getInputs(self, date, scoreTs=None, targetTs=None):
		if scoreTs == None:
			scoreTs = self.scoreTs
		if targetTs == None:
			targetTs = self.targetTs

		timedelta = datetime.timedelta(1)
		return tuple(itertools.chain.from_iterable((ts.pastNVals(date - timedelta, self.historySize) for ts in (scoreTs, targetTs))))

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
	def train(self, maxEpochs=1000):
		logit.debug("Number of training patterns: %i" % len(self.trainDs))
		logit.debug("Input and output dimensions: %i, %i" % (self.trainDs.indim, self.trainDs.outdim))
		logit.debug("First sample (input, target):")
		logit.debug(str((self.trainDs['input'][0], self.trainDs['target'][0])))
		
		logit.debug('Creating Trainer')
		trainer = BackpropTrainer(self.net, dataset=self.trainDs, verbose=True)

		logit.debug('Training Neural Net')
		trainer.trainUntilConvergence(maxEpochs=maxEpochs)
		logit.debug('Finished Training Neural Net')
		return self.net

	def unscale(self, val):
		return self.targetOffset + val / self.targetScalingFactor

	def predict(self, date, scoreTs=None, targetTs=None):
		return self.unscale(self.net.activate(self.getInputs(date, scoreTs=scoreTs, targetTs=targetTs))[0])

	# Uses the trained neural net to predict the stock in the given date range and returns a list of predicted values
	def predictTs(self, dateList, scoreTs=None, targetTs=None):
		return TimeSeries({ date: self.predict(date, scoreTs=scoreTs, targetTs=targetTs) for date in dateList })

	def getTrainingStats(self, plot=False):
		targetX, targetY = zip(*self.targetTs.getItems())
		targetY = [self.unscale(y) for y in targetY]
		actualTargetTs = self.targetTs.mapValues(self.unscale)

		trainingDates, testDates = self.getTrainAndTestDateLists()

		if plot:
			# Configure plots
			fig, ax = plt.subplots()
			y_formatter = ScalarFormatter(useOffset=False)
			ax.yaxis.set_major_formatter(y_formatter)
			plt.subplot(211)
			plt.plot(targetX, targetY, 'r', label='Targets')
			plt.legend()

		for dateList, color, label in ((trainingDates, 'g', 'Training'), (testDates, 'b', 'Testing')):
			predictedTs = self.predictTs(dateList)
			error, mse, mape, dirAcc = self.getStats(actualTargetTs, dateList=dateList, predictedTs=predictedTs)

			logit.info('%s MSE: %f' % (label, mse))
			logit.info('%s MAPE: %f' % (label, mape))
			logit.info('%s Directional Accuracy from actual: %f' % (label, dirAcc))

			if plot:
				# Prediction plot
				plt.subplot(211)
				plt.plot(dateList, predictedTs.getValueList(), color, label=label)

				# Error plot
				plt.subplot(212)
				plt.fill_between(dateList, error, facecolor=color)

	def getStats(self, actualTargetTs, dateList=None, predictedTs=None):
		if dateList == None:
			dateList = actualTargetTs.getDateList()
		if predictedTs == None:
			predictedTs = self.predictTs(dateList)

		error = [float(predictedTs.getVal(date) - actualTargetTs.getVal(date)) for date in dateList]

		mse = sum((e**2 for e in error)) / len(dateList)

		mape = 0
		avg = sum([actualTargetTs.getVal(date) for date in dateList]) / len(dateList)
		for date in dateList:
			divisor = actualTargetTs.getVal(date)
			if divisor == 0:
				divisor = avg
			mape += abs(float(predictedTs.getVal(date) - actualTargetTs.getVal(date)) / divisor)
		mape *= 100.0 / len(dateList)

		correct = 0
		for date in dateList[1:]:
			today, yesterday = predictedTs.pastNVals(date, 2)
			todayTarget, yesterdayTarget = actualTargetTs.pastNVals(date, 2)
			if (today - yesterdayTarget) * (todayTarget - yesterdayTarget) > 0:
				correct += 1
		dirAcc = float(correct) / (len(dateList) - 1)

		return error, mse, mape, dirAcc


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
		self.wordCounterTsFilename = wordCounterTsFilename

	def generateNeuralNet(self, stock, loadDataSetFromFile=False, randomScoreTs=False, maxEpochs=1000):
		logit.debug('Downloading Stock Data')
		targetTs = loadStockTs(stock, self.startDate, self.endDate)
		scoreTs = loadScoreTs(stock, wordCounterTsFilename=self.wordCounterTsFilename)
		if randomScoreTs:
			scoreTs = scoreTs.mapValues(lambda val: random.random())

		scaledInputTs, offset, scale = scoreTs.normalize()
		scaledTargetTs, offset, scale = targetTs.normalize()

		createNet = lambda historySize: createNLayerFFNet(historySize, 5, 10)
		nn = NeuralNet(createNet, 3, scaledInputTs, scaledTargetTs, targetScalingFactor=scale, targetOffset=offset, loadDataSetFromFile=loadDataSetFromFile)
		nn.train(maxEpochs=maxEpochs)

		return nn

def loadNN(filename=NN_FILENAME):
	with open(filename, 'rb') as f:
		nn = pickle.load(f)
	return nn

def saveNN(nn, filename=NN_FILENAME):
	with open(filename, 'wb') as f:
		pickle.dump(nn, f)

def getIdStr(sentIdStr, stock):
	return 'nn-%s-%s' % (stock, sentIdStr)

def genAllNNs(trainingStartDate, trainingEndDate, plot=False):
	logit.info('Begin training NNs')
	stocks = ['intc', 'aapl', 'msft', 'goog', 'dell'] # , 'fb', 'twtr'
	files = ['ts-greater-than-10-with-score-mult.p', 'ts-greater-than-10-score.p', 'ts-pos-score.p', 'ts-with-score-multiplier.p', 'ts.p'] #, 'ts-pg-only-with-score-mult.p', 'ts-pg-only.p', 'ts-greater-than-50-score.p', 'ts-greater-than-100-score.p']

	# for filename in files:
	filename = 'ts-greater-than-10-score.p'
	for stock in stocks:
		idStr = getIdStr(filename, stock)
		logit.info('idStr: %s' % idStr)

		net = StockNeuralNet(trainingStartDate, trainingEndDate, wordCounterTsFilename=filename)
		nn = net.generateNeuralNet(stock, loadDataSetFromFile=False)
		nn.getTrainingStats(plot=plot)

		saveNN(nn, filename=idStr)

		if plot:
			fig = plt.gcf()
			fig.canvas.set_window_title('stock %s, file %s' % (stock, filename))
			plt.draw()

	if plot:
		plt.show()

def testGeneratedNNs(startDate, endDate):
	logit.info('Begin testing pre-generated NNs')
	stocks = ['intc', 'aapl', 'msft', 'goog', 'dell']
	files = ['ts-greater-than-10-with-score-mult.p', 'ts-greater-than-10-score.p', 'ts-pos-score.p', 'ts-with-score-multiplier.p', 'ts.p'] #, 'ts-pg-only-with-score-mult.p', 'ts-pg-only.p', 'ts-greater-than-50-score.p', 'ts-greater-than-100-score.p']

	# for filename in files:
	filename = 'ts-greater-than-10-score.p'
	for stock in stocks:
		idStr = getIdStr(filename, stock)
		logit.info('idStr: %s' % idStr)

		nn = loadNN(filename=idStr)

		# nn.getStats(targetTs)


if __name__ == '__main__':
	genAllNNs(datetime.date(2011, 1, 1), datetime.date(2011, 12, 31))
	# testGeneratedNNs(datetime.date(2012, 1, 1), datetime.date(2012, 12, 31))
