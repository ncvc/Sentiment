import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from NeuralNet import loadNN, getIdStr, NeuralNet
from StockData import loadStockTs
from ScoreData import loadScoreTs
from TimeSeries import TimeSpan
from Preprocess import MultiTopicWordCounterTs


class Sim:
	def __init__(self, startDate, endDate, trader, stock):
		self.stockTs = loadStockTs(stock, startDate, endDate)
		self.scoreTs = loadScoreTs(stock)
		self.trader = trader
		self.amountStockOwned = 0

		dates = self.stockTs.getDateList()
		self.timeSpan = TimeSpan(dates[5], dates[-1])

	def run(self, plot=False):
		dateList = []
		amountStockOwnedList = []
		currentFundsList = []
		amountBoughtList = []

		for date in self.timeSpan:
			if self.trader.currentFunds <= 0:
				break

			if date in self.stockTs:
				dateList.append(date)
				amountStockOwnedList.append(self.amountStockOwned)
				currentFundsList.append(self.trader.currentFunds)

				amountBought = self.trader.trade(date, self.stockTs.getSlice(date), self.scoreTs.getSlice(date))
				success = self.executeTransaction(amountBought)

				amountBoughtList.append(amountBought)

			self.nextDay(date)

		print 'Started with $%f, ended with $%f' % (self.trader.initialFunds, self.trader.currentFunds)

		if plot:
			fig, ax = plt.subplots()
			y_formatter = ScalarFormatter(useOffset=False)
			ax.yaxis.set_major_formatter(y_formatter)
			plt.subplot()
			plt.plot(dateList, amountStockOwnedList, 'r', label='Amount Stock Owned')
			plt.plot(dateList, currentFundsList, 'b', label='Current Funds')
			plt.plot(dateList, amountBoughtList, 'g', label='Amount Bought')
			plt.legend()

			plt.show()

	def executeTransaction(self, amountBought):
		# Transaction failure conditions
		if amountBought > 0:
			if amountBought > self.trader.currentFunds:
				return False
		if amountBought < 0:
			if self.amountStockOwned < -amountBought:
				return False

		self.amountStockOwned += amountBought
		self.trader.currentFunds -= amountBought
		return True

	def nextDay(self, date):
		todayVal, yesterdayVal = self.stockTs.pastNVals(date, 2)
		self.amountStockOwned *= float(todayVal) / yesterdayVal


class Trader(object):
	def __init__(self, initialFunds):
		self.initialFunds = self.currentFunds = initialFunds

	def trade(self, date, stockTs, scoreTs):
		raise NotImplementedError


class AllInTrader(Trader):
	def trade(self, date, stockTs):
		return self.currentFunds


class NNTrader(Trader):
	def __init__(self, initialFunds, nn):
		super(NNTrader, self).__init__(initialFunds)
		self.nn = nn

	def trade(self, date, stockTs, scoreTs):
		currentVal = stockTs.pastNVals(date - datetime.timedelta(1), 1)[0]
		nextPredictedVal = self.nn.predict(date, scoreTs=scoreTs, targetTs=stockTs)

		predictedDelta = nextPredictedVal - currentVal
		toTrade = 10 if self.currentFunds >= 10 else self.currentFunds
		if predictedDelta > 0:
			return toTrade
		elif predictedDelta < 0:
			return -toTrade

		return 0


if __name__ == '__main__':
	nn = loadNN(getIdStr('ts-greater-than-10-score.p', 'msft'))
	trader = NNTrader(1000, nn)
	sim = Sim(datetime.date(2011, 1, 1), datetime.date(2011, 12, 31), trader, 'msft')
	sim.run(plot=True)
