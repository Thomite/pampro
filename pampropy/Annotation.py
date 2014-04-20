
class Annotation(object):

	def __init__(self, label, start_timestamp, end_timestamp):
		
		self.label = label
		self.start_timestamp = start_timestamp
		self.end_timestamp = end_timestamp
		self.draw_properties = {}

