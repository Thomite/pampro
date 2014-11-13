
import sys
from PyQt4 import QtGui
import random
import time
import math
from datetime import datetime, timedelta
import os

pampro_red = "#FA1010"
pampro_green = "#10FA10"
pampro_blue = "#1010FA"
pampro_darkgrey = "#101010"
pampro_lightgrey = "#FAFAFA"

pampro_background = "#FFFFFF"

all_style = """

/*QProgressBar { border: 1px solid """+pampro_darkgrey+"""; background-color:#FFFFFF; }
QProgressBar::chunk { background-color: """+pampro_red+"""; width: 1px; }*/

QMenu { background-color: """+pampro_background+"""; border: 1px solid """+pampro_darkgrey+"""; }
QMenu::item:selected { background-color: """+pampro_red+"""; }
QWidget { background-color: """+pampro_background+"""}
QMainWindow { background-color: """+pampro_background+"""}

QMenuBar {font-size:14px; background-color:"""+pampro_background+"""; border-bottom:1px solid #000000;}
QMenuBar::item { padding-right:50px; padding-left:5px; background: transparent; }
QMenuBar::item::selected {background-color: """+pampro_red+"""; }

QLineEdit {padding:5px; border:1px solid"""+pampro_darkgrey+"""; height:45px;}
QPushButton {padding:5px; border:1px solid"""+pampro_darkgrey+"""; height:45px;}
QSpinBox {padding:5px; border:1px solid"""+pampro_darkgrey+"""; height:45px;}
"""

def tohex(r,g,b):

	return "#" + str(hex(r))[2:] + str(hex(g))[2:] + str(hex(b))[2:]





class GUI(QtGui.QMainWindow):

	def __init__(self):

		super(GUI, self).__init__()

		self.setGeometry(0, 0, 325, 170)
		self.setWindowTitle('PAMPRO')
		self.setWindowIcon(QtGui.QIcon('temp.png'))

		grid = QtGui.QGridLayout()
		grid.setSpacing(10)

		self.file_dialog = QtGui.QFileDialog(self)

		self.button1 = QtGui.QPushButton('...', self)
		grid.addWidget(self.button1, 1, 3, 1, 1)

		self.button2 = QtGui.QPushButton('...', self)
		grid.addWidget(self.button2, 2, 3, 1, 1)

		self.button3 = QtGui.QPushButton('Run', self)
		grid.addWidget(self.button3, 4, 1, 1, 3)

		self.label1 = QtGui.QLabel(self)
		self.label1.setText("Job file:")
		
		grid.addWidget(self.label1, 1, 1, 1, 1)
	
		self.label2 = QtGui.QLabel(self)
		self.label2.setText("Script:")
		grid.addWidget(self.label2, 2, 1, 1, 1)

		self.label3 = QtGui.QLabel(self)
		self.label3.setText("Number of jobs:")
		grid.addWidget(self.label3, 3, 1, 1, 1)

		self.edit1 = QtGui.QLineEdit(self)
		grid.addWidget(self.edit1, 1, 2, 1, 1)

		self.edit2 = QtGui.QLineEdit(self)
		grid.addWidget(self.edit2, 2, 2, 1, 1)

		self.edit3 = QtGui.QSpinBox(self)
		self.edit3.setValue(1)
		self.edit3.setRange(1, 100000)
		grid.addWidget(self.edit3, 3, 2, 1, 1)
		

		self.edit1.setToolTip('Select the job file containing details of all files to be processed.')
		self.edit2.setToolTip('Select the script to be executed for each file.')
		self.edit3.setToolTip('Select the number of processes to perform this job.')

		# Actions
		action1 = QtGui.QAction(QtGui.QIcon('Capture.png'), 'Action1', self)
		action1.setShortcut('Ctrl+Q')
		action1.triggered.connect(self.blah)

		action2 = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Action2', self)
		action2.setShortcut('Ctrl+A')
		action2.triggered.connect(self.blah)

		run_action = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Run', self)
		run_action.setShortcut('Ctrl+R')
		run_action.triggered.connect(self.run)

		jobfile_action = QtGui.QAction(QtGui.QIcon('exit24.png'), 'blah', self)
		#jobfile_action.setShortcut('Ctrl+R')
		jobfile_action.triggered.connect(self.get_job_file)

		self.button1.clicked.connect(self.get_job_file)
		self.button2.clicked.connect(self.get_script)
		self.button3.clicked.connect(self.run)

		holder = QtGui.QWidget(self)
		holder.move(0,25)
		holder.setLayout(grid)
		holder.resize(holder.sizeHint())

		self.setStyleSheet(all_style)
		self.center()
		self.show()

	def blah(self):
		print "Hello!"


	def center(self):

		qr = self.frameGeometry()
		cp = QtGui.QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def refresh(self):

		pass

	def get_job_file(self):
		filename = self.file_dialog.getOpenFileName()
		self.edit1.setText(filename)


	def get_script(self):
		filename = self.file_dialog.getOpenFileName()
		self.edit2.setText(filename)

	def run(self):

		job_file = self.edit1.displayText()
		script = self.edit2.displayText()
		num_jobs = int(self.edit3.cleanText())

		if num_jobs > 1:
			print "Batch job"

			for i in range(1, num_jobs+1):
				command = "sge ~/anaconda/bin/ipython --matplotlib=None " + script + " " + job_file + " " + str(i) + " " + str(num_jobs)
				print command
				#os.system(command)
		else:
			print "Single process"

			command = "sge ~/anaconda/bin/ipython --matplotlib=None " + script + " " + job_file + " 1 1"
			print command
			#os.system(command)
		

		


if __name__ == '__main__':

	app = QtGui.QApplication(sys.argv)
	m = GUI()

	sys.exit(app.exec_())
