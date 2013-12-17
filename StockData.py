from TimeSeries import TimeSeries
import os
import csv
import urllib
from datetime import datetime, date


STOCK_DATA_FOLDER = 'stock_data'


# Loads stock data from file into a time series
# TODO: This is a terrible way to store the data but it works for now
def loadStockTs(stock, startDate, endDate):
	filepath = os.path.join(STOCK_DATA_FOLDER, stock + '[%s,%s].csv' % (startDate, endDate))

	if not os.path.exists(filepath):
		if not os.path.exists(STOCK_DATA_FOLDER):
			os.makedirs(STOCK_DATA_FOLDER)

		params = urllib.urlencode({    # Below is literally the worst API design. Classic Yahoo.
			's': stock,
			'a': startDate.month - 1,  # WHY
			'b': startDate.day,
			'c': startDate.year,
			'd': endDate.month - 1,    # WHAT IS WRONG WITH YOU, YAHOO
			'e': endDate.day,
			'f': endDate.year,
			'g': 'd',
			'ignore': '.csv',          # WHAT COULD THIS POSSIBLY MEAN
		})
		url = 'http://ichart.yahoo.com/table.csv?%s' % params
		try:
			urllib.urlretrieve(url, filepath)
		except urllib.ContentTooShortError as e:
			with open(filepath, "w") as outfile:
				outfile.write(e.content)

			print 'Error retrieving stock %s' % stock
			return

	with open(filepath) as f:
		ts = TimeSeries({ datetime.strptime(row['Date'], '%Y-%m-%d').date(): float(row['Close']) for row in csv.DictReader(f) })
	return ts

if __name__ == '__main__':
	print loadStockTs('msft', date(2011,1,3), date(2011,1,3))
