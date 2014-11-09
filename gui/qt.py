
import sys
from PyQt4 import QtGui
import random
import time
import math
from datetime import datetime, timedelta

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
"""

def tohex(r,g,b):

    return "#" + str(hex(r))[2:] + str(hex(g))[2:] + str(hex(b))[2:]



class PAMPRO_Process():

    def __init__(self, monitor):

        self.last_updated = datetime.now()

        self.pbar = QtGui.QProgressBar(monitor)
        self.pbar.setTextVisible(False)

        val = random.randint(0,100)
        self.pbar.setValue(val)
        self.pbar.setToolTip(str(val))
        self.pbar.setFixedSize(25,25)
        #self.pbar.clicked.connect(monitor.blah)

    def refresh(self):
        pass

        # self.pbar.setToolTip('Tooltip for progress bar')


class Monitor(QtGui.QMainWindow):

    def __init__(self):

        super(Monitor, self).__init__()

        self.setGeometry(0, 0, 400, 400)
        self.setWindowTitle('PAMPRO Monitor')
        self.setWindowIcon(QtGui.QIcon('temp.png'))

        grid = QtGui.QGridLayout()
        grid.setSpacing(2)


        # Create a button
        #button = QtGui.QPushButton('Button 1', self)
        #button.setToolTip('Tooltip for button1')
        #grid.addWidget(button, 2, 1, 1, 1)

        # Actions
        action1 = QtGui.QAction(QtGui.QIcon('Capture.png'), 'Action1', self)
        action1.setShortcut('Ctrl+Q')
        action1.triggered.connect(self.blah)

        action2 = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Action2', self)
        action2.setShortcut('Ctrl+A')
        action2.triggered.connect(self.blah)


        # Create list of processes
        num_processes = 100
        x,y = 1,1
        max_x = math.floor(math.sqrt(num_processes))
        self.processes = []
        for i in range(num_processes):
            p = PAMPRO_Process(self)
            self.processes.append(p)
            grid.addWidget(p.pbar, x, y, 1, 1)
            if x == max_x:
                x = 1
                y += 1
            else:
                x += 1

        # Menu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(action1)
        fileMenu.addAction(action2)
        fileMenu2 = menubar.addMenu('Preferences')
        fileMenu2.addAction(action1)
        fileMenu2.addAction(action2)

        holder = QtGui.QWidget(self)
        holder.move(0,25)
        holder.setLayout(grid)
        holder.resize(holder.sizeHint())

        self.setStyleSheet(all_style)
        self.center()
        self.show()

    def blah(self):
        print "Hello!"
        for p in self.processes:
            p.pbar.setValue(p.pbar.value()+1)

    def center(self):

        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def refresh(self):

        for p in self.processes:
            p.refresh()
        print("Refreshing")

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    m = Monitor()

    sys.exit(app.exec_())
