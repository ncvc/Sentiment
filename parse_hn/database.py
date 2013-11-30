import datetime

from peewee import MySQLDatabase, Model, CharField, DateTimeField, IntegerField, BooleanField, TextField

database = MySQLDatabase('hn', host='localhost', port=3306, user='root', passwd='')

# Database must use utf8mb4 for smileys and other such nonesense
# ALTER DATABASE hn CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;


# Model definitions
class BaseModel(Model):
	class Meta:
		database = database

class Story(BaseModel):
	id = IntegerField(primary_key=True)
	score = IntegerField(null=True)
	kids = TextField(null=True)
	author = TextField(null=True)
	text = TextField(null=True)
	type = CharField(null=True)
	parent = IntegerField(null=True)
	time = DateTimeField(formats='%Y-%m-%d %H:%M:%S', null=True)
	url = TextField(null=True)
	title = TextField(null=True)
	dead = BooleanField(null=True)
	deleted = BooleanField(null=True)
	parts = TextField(null=True)


# Handles all database operations
class DB:
	def __enter__(self):
		database.connect()
		database.execute_sql('SET NAMES utf8mb4;')  # Necessary for some emojis
		return self

	def __exit__(self, type, value, traceback):
		print 'DB.__exit__', type, value, traceback
		database.close()

	# Simple utility function to create tables
	def create_tables(self):
		Story.create_table()

	# Get story by id
	def get_story(self, id):
		return Story.get(Story.id == id)

	# Get all comments in the date range
	def get_comments(self, startDate, endDate):
		return Story.select().where((Story.type == 'comment') & (Story.time.between(startDate, endDate)))

	# Adds the story data to the db
	def add_story(self, storyData):
		story = Story()

		story.id = storyData.get('id')
		story.kids = storyData.get('kids')
		story.author = storyData.get('by')
		story.text = storyData.get('text')
		story.type = storyData.get('type')
		story.parent = storyData.get('parent')
		story.url = storyData.get('url')
		story.title = storyData.get('title')
		story.dead = storyData.get('dead')
		story.deleted = storyData.get('deleted')
		story.parts = storyData.get('parts')

		score = storyData.get('score')
		if isinstance(score, (int, long)):
			story.score = score
		else:
			story.score = 0

		time = storyData.get('time')
		if time == None:
			story.time = None
		else:
			story.time = datetime.datetime.fromtimestamp(1160494499)

		# Write the new row to the database
		story.save(force_insert=True)


if __name__ == '__main__':
	with DB() as db:
		db.create_tables()
