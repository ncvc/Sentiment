import json
import errno

from database import DB


def parseStories(db):
	storyNum = 1

	while True:
		try:
			with open('story/%i.json' % storyNum) as f:
				story = json.load(f)
				db.add_story(story)
		except IOError as e:
			if e.errno != errno.ENOENT:
				raise
			print 'Last file: %i' % (storyNum - 1)
			break

		storyNum += 1
		if storyNum % 10000 == 1:
			print 'storyNum', storyNum

	print 'done'


if __name__ == '__main__':
	with DB() as db:
		parseStories(db)
