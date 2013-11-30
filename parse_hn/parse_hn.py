import json
import errno
import time

from database import DB

# Last file: 6793246


def parseStories(db):
	storyNum = 1
	errCount = 0
	start = mid = time.time()

	while True:
		try:
			with open('/home/ncvc/hn_json/story/%i.json' % storyNum) as f:
				story = json.load(f)
				db.add_story(story)

				errCount = 0
		except IOError as e:
			if e.errno != errno.ENOENT:
				raise
			print 'Last file: %i' % (storyNum - 1)

			errCount += 1
			if errCount >= 100:
				break

		storyNum += 1
		if storyNum % 10000 == 1:
			print 'storyNum', storyNum, time.time() - mid
			mid = time.time()

	print 'done', time.time() - start


if __name__ == '__main__':
	with DB() as db:
		parseStories(db)
