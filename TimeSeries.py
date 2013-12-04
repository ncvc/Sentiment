import datetime


class TimeSeries:
	def __init__(self, tsDict=None):
		if tsDict == None:
			self.ts = {}
		else:
			self.ts = tsDict

	def getMinValue(self):
		return min(self.ts.itervalues())

	def getMaxValue(self):
		return max(self.ts.itervalues())

	def pastNVals(self, startDate, n):
		timedelta = datetime.timedelta(1)

		currentDate = startDate
		numReturned = 0
		vals = [0] * n
		while numReturned < n:
			if currentDate in self.ts:
				vals[numReturned] = self.ts[currentDate]
				numReturned += 1
			currentDate -= timedelta

		return vals

	def getVal(self, date):
		return self.ts[date]

	def getItems(self):
		return sorted(self.ts.items())

	def getValueList(self):
		return [val for date, val in sorted(self.ts.items())]

	def mapValues(self, function):
		return TimeSeries({date: function(value) for date, value in self.ts.items()})


def dateIterator(startDate, endDate, backward=False):
	currentDate = startDate

	if backward:
		step = -1
	else:
		step = 1
	timedelta = datetime.timedelta(step)

	while currentDate != endDate:
		yield currentDate
		currentDate += timedelta
