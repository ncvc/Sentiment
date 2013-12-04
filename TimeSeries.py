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

	def getValueList(self):
		return [val for date, val in sorted(self.ts.items())]

	def mapValues(self, function):
		return TimeSeries({date: function(value) for date, value in sorted(self.ts.items())})
