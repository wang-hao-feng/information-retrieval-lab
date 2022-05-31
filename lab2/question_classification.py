import jieba
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

jieba.setLogLevel(logging.INFO)

class QuestionClassifier():
    def __init__(self, iter=100, C=5000, stop_words=[]) -> None:
        self.class2cid = {}
        self.cid2class = {}
        self.wh = ['都有哪些', '什么地方', '什么内容', '什么时候',
                 '哪三段', '为什么', '怎么样', '哪一部', '多少个', '多少钱', '哪个人', '什么样', '哪一年', '哪一天',
                 '什么', '哪里', '哪儿', '几个', '如何', '几层', '哪年', '多少', '怎么', '哪些',
                 '何时', '几条', '哪个', '多重', '多长', '多大', '多久', '多宽', '多深', '多远',
                 '哪天', '是谁', '时间', '第几', '谁', '哪', '几', '吗', '多']
        self.__word2vec = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=stop_words)
        self.__classifier = LogisticRegression(max_iter=iter, C=C, solver='liblinear')
    
    def train(self, train_path:str) -> None:
        train_X, train_Y = self.GetDataset(train_path)

        #获取类和序号的映射表
        classes = set(train_Y)
        for c in classes:
            self.class2cid[c] = len(self.class2cid)
            self.cid2class[len(self.cid2class)] = c
        #将类替换为序号
        train_Y = [self.class2cid[c] for c in train_Y]

        #训练tf-idf
        train_X = self.__word2vec.fit_transform(train_X).toarray()

        #训练分类器
        self.__classifier.fit(train_X, train_Y)

    def GetDataset(self, dataset_path:str) -> tuple:
        """
        读取数据集并整理
        """
        #读取数据
        with open(dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        dataset = [line.split() for line in lines]
        
        for word in self.wh:
            jieba.add_word(word)
        dataset = [data for data in dataset if data[0][:3] not in ['UNK']]

        return [' '.join(jieba.cut(data[1])) for data in dataset], [data[0][:3] for data in dataset]

    def Sort(self, questions:list) -> str:
        questions = self.__word2vec.transform(questions)
        ans = list(self.__classifier.predict(questions))
        return [self.cid2class[cid] for cid in ans]

if __name__ == '__main__':
    #测试分类器
    classifier = QuestionClassifier()
    classifier.train('./question_classification/trian_questions.txt')

    f1 = {'HUM':None, 'LOC':None, 'NUM':None, 'TIM':None, 'OBJ':None, 'DES':None}
    test_X, test_Y= classifier.GetDataset('./question_classification/test_questions.txt')
    ans = classifier.Sort(test_X)
    test_Y = np.array([classifier.class2cid[c] for c in test_Y])
    ans = np.array([classifier.class2cid[c] for c in ans])
    for c in f1:
        cid = classifier.class2cid[c]
        a = (test_Y == cid)
        b = (ans == cid)
        p = np.sum(a * b) / np.sum(a)
        r = np.sum(a * b) / np.sum(b)
        f1[c] = 2 * p * r / (p + r)
    print('f1:{0}'.format(f1))
