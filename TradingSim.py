from NeuralNet import loadNN, getIdStr
from StockData import loadStockData
from TimeSeries import TimeSpan


class Sim:
	def __init__(self, startDate, endDate, trader):
		self.timeSpan = TimeSpan(startDate, endDate)
		self.stockTs = loadStockData(startDate, endDate)
		self.trader = trader

	def run(self):
		for date in self.timeSpan:
			if self.trader.currentFunds <= 0:
				break

			self.trader.processData(date, newData)
			self.trader.trade(date, self.stockTs.getSlice(date))

		print 'Started with $%f, ended with $%f' % (self.trader.initialFunds, self.trader.currentFunds)


class Trader:
	def __init__(self, initialFunds):
		self.initialFunds = self.currentFunds = initialFunds

	def trade(self, date, stockTs):
		raise NotImplementedError

	def processData(self, date, newData):
		pass


class AllInTrader(Trader):
	def trade(self, date, stockTs):
		return self.currentFunds


class NNTrader(Trader):
	def __init__(self, initialFunds, nn):
		super().__init__(initialFunds)
		self.nn = nn
		self.scaledScoreTs = nn.prepareScoreTs(scoreTs)

	def trade(self, date, stockTs):
		self.nn.predict(date, scoreTs=self.scaledScoreTs, targetTs=stockTs)

	def processData(self, date, newData):
		# Analyze sentiment of the new day's data
		# Insert data into scoreTs
		pass


if __name__ == '__main__':
	nn = loadNN(getIdStr('ts-greater-than-10-score.p', 'msft'))
	trader = NNTrader(1000, nn)
	sim = Sim()
