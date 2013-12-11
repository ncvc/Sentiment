import datetime


# Used to represent a time series. Internally represented as a dictionary where the keys are datetime.date objects
# and the values are the values at the corresponding date
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

	def getSlice(self, endDate):
		return TimeSeries({ date: val for date, val in self.ts.items() if date < endDate })

	def getVal(self, date):
		return self.ts[date]

	def getItems(self):
		return sorted(self.ts.items())

	def getValueList(self):
		return [val for date, val in self.getItems()]

	def mapValues(self, function):
		return TimeSeries({date: function(value) for date, value in self.ts.items()})

	def getDeltaTs(self):
		return TimeSeries({date: (value - self.pastNVals(date - datetime.timedelta(1), 1)[0]) for date, value in self.getItems()[1:]})

	def getDateList(self):
		return sorted(self.ts.keys())

	# Scale data linearly from [0,1]
	def normalize(self):
		minVal = self.getMinValue()
		maxVal = self.getMaxValue()
		try:
			scale = 1.0 / (maxVal-minVal)
		except ZeroDivisionError:
			scale = 1

		return self.linearTransform(minVal, scale), minVal, scale

	def linearTransform(self, offset, scale):
		return self.mapValues(lambda val: float(val-offset) * scale)


class TimeSpan:
	def __init__(self, startDate, endDate):
		self.startDate = startDate
		self.endDate = endDate

	def __iter__(self):
		return dateIterator(self.startDate, self.endDate)


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


if __name__ == '__main__':
	span = TimeSpan(datetime.date(2013, 1, 27), datetime.date(2013, 2, 3))
	for date1 in span:
		for date2 in span:
			print date1, date2
