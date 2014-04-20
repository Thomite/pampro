
class Annotation(object):

	def __init__(self, label, start_timestamp, end_timestamp):
		
		self.label = label
		self.start_timestamp = start_timestamp
		self.end_timestamp = end_timestamp
		self.draw_properties = {'lw':0, 'alpha':0.7, 'facecolor':[0.78431,0.78431,0.196]}

def annotations_from_bouts(bouts):

	annotations = []
	for bout in bouts:

		a = Annotation("A", bout[0], bout[1])
		annotations.append(a)

	return annotations