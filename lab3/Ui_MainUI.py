# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\实验\信息检索\lab3\MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.SearchButtom = QtWidgets.QPushButton(self.centralwidget)
        self.SearchButtom.setGeometry(QtCore.QRect(610, 40, 93, 28))
        self.SearchButtom.setObjectName("SearchButtom")
        self.Result = QtWidgets.QListWidget(self.centralwidget)
        self.Result.setGeometry(QtCore.QRect(25, 91, 751, 441))
        self.Result.setObjectName("Result")
        self.LastPage = QtWidgets.QPushButton(self.centralwidget)
        self.LastPage.setGeometry(QtCore.QRect(30, 540, 93, 28))
        self.LastPage.setObjectName("LastPage")
        self.NextPage = QtWidgets.QPushButton(self.centralwidget)
        self.NextPage.setGeometry(QtCore.QRect(680, 540, 93, 28))
        self.NextPage.setObjectName("NextPage")
        self.Safe = QtWidgets.QComboBox(self.centralwidget)
        self.Safe.setGeometry(QtCore.QRect(50, 40, 101, 31))
        self.Safe.setMouseTracking(False)
        self.Safe.setObjectName("Safe")
        self.SearchBar = QtWidgets.QLineEdit(self.centralwidget)
        self.SearchBar.setGeometry(QtCore.QRect(170, 40, 431, 31))
        self.SearchBar.setObjectName("SearchBar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.Safe.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SearchButtom.setText(_translate("MainWindow", "Search"))
        self.LastPage.setText(_translate("MainWindow", "LastPage"))
        self.NextPage.setText(_translate("MainWindow", "NextPage"))
