import os
import sys
import json
import jieba
import logging
import numpy as np
from tqdm import tqdm
from Ui_MainUI import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication

jieba.setLogLevel(logging.INFO)

class GUI(QMainWindow, Ui_MainWindow):
    def __init__(self, k1=1.5, b=0.75):
        super().__init__()
        self.setupUi(self)

        self.SearchButtom.clicked.connect(self.__Search)
        self.LastPage.clicked.connect(self.__LastPage)
        self.NextPage.clicked.connect(self.__NextPage)
        self.Result.itemClicked.connect(self.__ClickedItem)
        self.Safe.addItems(['总经理', '经理', '大堂经理', '员工'])
        self.__level = {'员工':0, '大堂经理':1, '经理':2, '总经理':3}
        self.__result_per_page = 20

        self.__file_roots, documents, documents_len, self.__safe, self.__vocabulary = self.__ReadDocuments()

        self.__tf, self.__idf = self.__GetTfIdf(documents)

        self.__alpha = 1 - b + b * documents_len / np.average(documents_len)
        
        self.__k1 = k1

        self.__rank = None
        self.__now = {}
        self.__state = 0
    
    def __ReadDocuments(self):
        with open('./documents.json', 'r', encoding='utf-8') as file:
            lines = [json.loads(line) for line in file.readlines()]
        file_roots = []
        documents = []
        documents_len = []
        safe = []
        vocabulary = set()
        for line in lines:
            safe.append(line['safe'])
            document = line['title'] + line['parapraghs']
            file_root = os.path.join('../lab1/craw','_'.join(line['url'].split('/')[-4:]))
            file_roots.append(file_root)
            for file_name in line['file_name']:
                pass
            documents.append(document)
            documents_len.append(len(document))
            vocabulary |= set(document)
        return file_roots, documents, np.array(documents_len), safe, vocabulary

    def __GetTfIdf(self, documents:list):
        """
        计算tf和idf的值
        """
        documents_num = len(self.__file_roots)
        #初始化tf与df
        tf = [{} for _ in range(documents_num)]
        idf = dict.fromkeys(self.__vocabulary, 0)

        #遍历计算tf
        for pid in tqdm(range(documents_num), desc='calculating tf'):
            for word in documents[pid]:
                if word not in tf[pid]:
                    tf[pid][word] = 0
                tf[pid][word] += 1
            for word in set(documents[pid]):
                idf[word] += 1
        
        #计算idf
        for word in self.__vocabulary:
            idf[word] = np.log10(documents_num) - np.log10(idf[word])
        
        return tf, idf

    def __Search(self):
        query = self.SearchBar.text()
        level = self.__level[self.Safe.currentText()]

        query = [word for word in jieba.cut(query) if word in self.__vocabulary]
        documents_num = len(self.__file_roots)
        #计算rsv
        rsv = np.zeros(documents_num)
        tf = None
        for word in query:
            tf = np.zeros(documents_num)
            for pid in range(documents_num):
                tf[pid] = self.__tf[pid][word] if word in self.__tf[pid] else 0
            rsv += self.__idf[word] * (self.__k1 + 1) * tf / (self.__k1 * self.__alpha + tf)
        
        self.__rank = [rank for rank in np.argsort(-rsv).tolist() if self.__safe[rank] <= level]
        self.__state = 0
        self.__Show()

    def __LastPage(self):
        if self.__state > 0:
            self.__state -= 1
        else:
            return
        self.__Show()

    def __NextPage(self):
        if (self.__state + 1) * self.__result_per_page < len(self.__rank):
            self.__state += 1
        else:
            return
        self.__Show()

    def __Show(self):
        self.Result.clear()
        self.__now = {}
        head = self.__state * self.__result_per_page
        tail = min(head + self.__result_per_page, len(self.__rank))
        for id in self.__rank[head:tail]:
            with open(os.path.join(self.__file_roots[id], 'page_description.json'), 'r', encoding='utf-8') as file:
                title = json.load(file)['title']
            self.__now[title] = id
            self.Result.addItem(title)

    def __ClickedItem(self, item):
        path = self.__file_roots[self.__now[item.text()]]
        os.startfile(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())